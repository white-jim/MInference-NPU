# 上下文检查点 — MInference 1.0 → 昇腾 NPU 算法迁移

> 用途：在新会话中快速恢复工作上下文。详细信息在各专题文档里，本文件只列必读关键点。
> 最近更新：2026-05-26 — **路径 A 弃用，v2 转向 triton-ascend / tilelang-ascend 真稀疏 FA（PR-4）**。PR-3 把 8K prefill 拉到 7.28s（vs dense 1.95s，3.73×），但 16K 实测 21.95s vs dense 4.68s（4.69×，差距扩大不收敛），且 O(S²) bool mask 在 128K 必 OOM —— **路径 A 是死胡同**。用户升级到 CANN 8.5.0 + torch_npu 2.7.1.post4 + triton-ascend 3.2.0，并在 `flexhead-tl` 中跑通 tilelang 官方 `sparse_attention_fwd` 三个闸门 case：sanity / block-sparse / stream-llm 均 PASS（详见 §13.7）。当前仅需继续 A-shape 与 block-sparse 两种模式；剩余关键适配问题是 pad 不容错、`kv_group==1` 限制、标准 K/V 分离到官方 packed KV 语义的映射。
> 工作目录：`D:\tempt\算法迁移\`（不是 git repo；子目录 `MInference-NPU/` 是 git repo）

---

## 1. 任务一句话

把 Microsoft [MInference 1.0](https://github.com/microsoft/MInference) 长上下文 LLM 推理加速算法迁移到 **华为昇腾 NPU**，v1 走 **`npu_fusion_attention` + bool mask（路径 A）** 一条路（Triton-Ascend 留 v2，原因见 `MInference-NPU/docs/SETUP.md` §2：CANN 8.1.RC1 与 triton-ascend 版本矩阵不匹配），宿主 **HF transformers + torch_npu**。

- 上游源码（参考蓝本，**不动**）：`MInference/`
- v1 产出主线：`MInference-NPU/`
- 全部专题文档：`docs/`

---

## 2. v1 决策（已拍板，写在 `docs/migration_plan_v1.md`）

| 维度 | v1 选择 | v2 PR-4 修订（2026-05-26） |
|---|---|---|
| 迁移范围 | vertical_and_slash + block_sparse + stream_llm + dense fallback | 不变 |
| 排除项 | dilated / static / tri_shape / kvcompress / dist_ops / vLLM 集成 / FA3 / KV-CPU offload | 不变 |
| **算子主路径** | `npu_fusion_attention` + bool mask（路径 A） | **改走 triton-ascend 真稀疏 FA kernel（路径 B）**。路径 A 归档到 `minference/legacy/path_a/`，原因见 §13 |
| `convert_vertical_slash_indexes` | CPU Python 双指针（M4-a） | PR-3 已用 `_build_vs_mask_direct` 跳过；PR-4 进一步整体重写到 triton-ascend kernel 内部 |
| 框架宿主 | HF transformers + torch_npu；accelerate `device_map="auto"` 多卡 OK | 不变 |
| 产出形态 | 独立 `MInference-NPU/` 完整 py 源 + tests + examples + docs + 独立 setup.py | 不变 |

---

## 3. 进度

```
[x] 调研 + 方案 + 4 项决策拍板
[x] M0 环境与骨架                    详见 docs/SETUP.md
[x] M1 上层 Python 链路（dense fallback） 详见 docs/M1_dense_pipeline.md
[x] M2 Streaming kernel              详见 docs/M2_streaming.md
[x] M3 Block-Sparse kernel           详见 docs/M3_block_sparse.md
[x] M4 Vertical-Slash kernel（M4-a CPU 双指针 + M4-b NPU mask） 详见 docs/M4_vertical_slash.md
[x] Bug 修复 首轮（B1-B5/P3/P4，2026-05-24）
[x] Bug 修复 第二轮（B6-B9，2026-05-25）
[x] M5 端到端联调 + 精度/性能/效果报告（全部 kernel test + HF 端到端 8K/16K/32K 三方对比 + 128K 多卡 ALL DONE）
[x] docs/migration_v1_notes.md + docs/migration_v1_report.md（2026-05-25）
```

> 代码改造与实机校验解耦：Windows 工作机上写代码、syntax 检查通过即可推进；NPU 实机跑 `tests/test_*.py` 是用户拿到机器后并行做。

---

## 4. M5 入手要点（下一步直接接的事）

M5 不引入新代码，只做实机验证：

| 验证项 | 命令 |
|---|---|
| 环境烟测 | `python tests/test_env.py` |
| Dense pipeline | `python tests/test_dense_forward.py` |
| Streaming | `python tests/test_streaming_kernel.py` |
| Block-sparse | `python tests/test_block_sparse_kernel.py` |
| Vertical-slash | `python tests/test_vertical_slash_kernel.py` |
| HF 端到端 | `python examples/run_hf_minimal.py --attn_type minference ...` |

产出 `docs/migration_v1_notes.md`（实测踩坑）和 `docs/migration_v1_report.md`（精度/性能/效果三维对比）。

---

## 5. NPU v1 关键文件地图

| 角色 | 路径 |
|---|---|
| 顶层入口 | `MInference-NPU/minference/models_patch.py`（`MInference` 类） |
| HF monkey-patch | `MInference-NPU/minference/patch.py`（`minference_patch` / `patch_hf`） |
| Per-head 调度 + 顶层 forward | `MInference-NPU/minference/modules/minference_forward.py` |
| NPU dense 薄封装 | `MInference-NPU/minference/backend_npu/attention.py` |
| `convert_vertical_slash_indexes`（CPU Python） | `MInference-NPU/minference/backend_npu/cuda_shim.py` |
| Streaming kernel | `MInference-NPU/minference/ops/streaming_kernel_npu.py` |
| Block-Sparse kernel | `MInference-NPU/minference/ops/block_sparse_kernel_npu.py` |
| Vertical-Slash kernel | `MInference-NPU/minference/ops/vertical_slash_kernel_npu.py` |
| 上游参考（不 import，仅对照） | `MInference-NPU/minference/{patch,models_patch}_upstream.py` + `modules/minference_forward_upstream.py` |
| Best-pattern JSON | `MInference-NPU/minference/configs/*.json`（直接复用上游） |

上游对照路径（grep `MInference/` 子树即可，不在此处重复）。

---

## 6. MInference 1.0 算法关键点（迁移时反复用到）

详见 `docs/MInference_1.0_implementation.md`。**只摘必须记住的事实**：

- **加速对象**：仅 prefill；`q_len == 1` decode 短路成 dense flash_attn。
- **三种稀疏**：`vertical_and_slash`（主力，~95% head）、`block_sparse`、`stream_llm`（A-shape）。
- **best_pattern JSON**：每 (layer, head) 一组 `(pattern_type, v_size, s_size, score)`；常见默认 `vertical_and_slash, v=1000, s=6096`。
- **在线估计**（vertical_and_slash）：用最后 64 个 query 与全 K 做 QK，列和→topk→`v_idx`（升序）；反对角线和→topk→`s_idx`（降序）。强制保留前 30 列 sink + 最近 100 条 slash。
- **per-head 循环**：不同 head 选不同 pattern，无法 batch；本身开销远小于稀疏 kernel 收益。
- **starting_layer**：前若干层走 dense（pattern 未收敛）。
- **v1 不构建** 上游 `minference.cuda` CUDAExtension（`csrc/{kernels.cpp, vertical_slash_index.cu}`），由 `backend_npu/cuda_shim.py` 顶替。

---

## 7. 环境与守则

- **平台**：Windows 10 Pro / bash / Unix 风格路径；目录含中文「算法迁移」，shell 操作需引号包裹。
- **目标 NPU 服务器**（实机）：aarch64 / openEuler 22.03 LTS-SP4 / 910B3 / 驱动 25.0.rc1.1 / CANN Toolkit 8.1.RC1（`ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1`）/ Python 3.10 / torch 2.5.1 + torch_npu 2.5.1 / transformers 4.57.3 / accelerate 0.34.2。Triton-Ascend 在 CANN 8.1.RC1 下不作为默认依赖。完整矩阵见 `MInference-NPU/docs/SETUP.md`。
- **device-agnostic 守则**（M1 落地、M2-M4 守住）：
  1. 禁止写死 `npu:0` 或 `device="cuda"`，所有 tensor 跟随输入 device。
  2. 若 v2 引入 Triton-Ascend kernel，launch 同样需要跟随输入 device（v1 暂未启用，路径 A 全程走 `npu_fusion_attention`）。
  3. accelerate `device_map="auto"` 多卡靠这条覆盖；vLLM TP / 序列并行 / EP 留 v2。
- **长上下文多卡**：手动给 `max_memory` 留 KV 余量（配置层面，不改代码）。

---

## 8. 仍待处理（性能优化，留 M5 实测后做）

以下问题影响性能而非正确性，重构成本大且容易引入新错，**先看 M5 真实 profile 数据再判断要不要动**：

| 编号 | 文件 | 一句话 |
|---|---|---|
| P5 | `ops/vertical_slash_kernel_npu.py` + `backend_npu/cuda_shim.py` | vertical/slash 的 CPU Python 双指针 index convert 在 128k 上可能秒级；待向量化或 NPU kernel |
| P6 | M3/M4 kernel | token 级 `[B,H,S_q,S_k]` bool mask 16k 就重，32k+ 不实用；需真正的 block/index 稀疏 kernel |
| P7 | M3/M4 kernel | `S > 16384` silent dense fallback；M5 性能测试时若覆盖长上下文需手动检查 warning，避免误判 |
| P8 | `modules/minference_forward.py` | per-head 循环造成大量小 kernel launch；按 pattern 分组 batching 是优化空间 |
| P9 | `modules/minference_forward.py` | GQA/MQA 用 `repeat_kv` 物理复制，长上下文放大 KV 工作量 |

性能修复决策原则：M5 拿到真实瓶颈再动，不预先重构。

---

## 9. M5 实机踩坑记录

### 9.1 `npu_fusion_attention` 的 `sparse_mode` quirk（CANN 8.1.RC1 首发，8.2.RC1 仍存在）

**触发**：`python tests/test_env.py -v` 中 `test_npu_fusion_attention_smoke` FAIL，`max_abs_diff=3.633e+00`，差异跟"causal 期望 vs full attention 实测"对得上。

**实测组合**（B=1, N=4, S=256, D=128, fp16，与手写 PyTorch eager causal 比对）：

| 调用 | `max_abs_diff`（8.1.RC1） | `max_abs_diff`（8.2.RC1） | 实际语义 |
|---|---|---|---|
| `sparse_mode=0`（无 mask） | ~3.6 | ~3.6 | full attention |
| `sparse_mode=2`（无 mask） | ~3.6 | **3.633e+00** | full attention（**不是** causal） |
| `sparse_mode=2 + pre_tockens/next_tockens` | ~3.6 | — | full attention |
| `sparse_mode=4 + pre_tockens=W-1, next_tockens=0` | ~3.6 | **3.633e+00**（vs full=**2.441e-04**）| full attention（**不是** sliding-window band） |
| `sparse_mode=1 + 显式 [S_q, S_k] bool atten_mask` | ~0.00195（mean ~2e-5） | **1.953e-03** | 正确 causal |

**修正**：路径 A 全部 causal 调用改走 `sparse_mode=1 + 显式 bool atten_mask`（`True=masked`，与 M3/M4 既有约定一致），`try/except TypeError` 兜底 uint8。已修 `backend_npu/attention.py::dense_attention` + `tests/test_env.py` + `docs/SETUP.md` §5/§5.1。修正后实机 smoke max_abs_diff=1.953e-03，PASS。

**CANN 8.2.RC1 复测结论（2026-05-25 由 `tests/test_sparse_mode_quirk.py` 实测）**：
- sm2 仍退化为 full attention（与 8.1 同症）
- sm4 同样退化为 full attention，且 `out_npu` 与 full attention 输出 `max_abs=2.441e-04` 几乎 bit-identical，**确证 `pre_tockens / next_tockens` 在 8.2.RC1 下被无视**
- baseline sm1 + 显式 mask 仍精度正确
- v2 路线确定为**分支 B**：O(S²) mask 必须留着，CANN 升级无法自动解决 128K OOM

**仍需盯**：CANN 升到 8.3+ 时可再复测，命中后回滚省 mask 显存（届时跑 `tests/test_sparse_mode_quirk.py` 即可）。在 8.2.RC1 上：v1 现有显式 mask 路线零回归（M1~M5 单测 93 passed），不必改。

### 9.2 transformers 4.57.3 attention forward 签名变更（2026-05-25）

**触发**：`tests/test_dense_forward.py` 3/3 FAIL，错误 `forward() missing 2 required positional arguments: 'past_key_value' and 'output_attentions'`。

**根因**：transformers 4.57.3 `LlamaAttention.forward` 签名重写：
- 位置 2 由 `attention_mask` 改为 `position_embeddings`
- `past_key_value` → `past_key_values`（复数）
- 删除 `output_attentions`（layer 不再传给 attention）
- 返回值由三元组改为二元组 `(attn_output, attn_weights)`
- `LlamaDecoderLayer` 改用全 kwargs 调 `self_attn`
- `num_heads` / `num_key_value_heads` 不再挂在 attention module 实例上，需从 `self.config` 读

**修正**：`minference/modules/minference_forward.py::forward` 所有参数改默认值，名字对齐新签名，加 `cache_position` / `position_embeddings` 命名参数，返回 2-tuple；`past_key_value`（单数）旧版通过 kwargs 兜底。

**测试 3 副作用**：`accelerate.dispatch_model` 在 NPU 上 AlignDevicesHook 没把 CPU input_ids 自动搬到 embed 卡 → 测试侧改成显式 `.to(embed_device)`。另外 `infer_auto_device_map` 在 8×NPU + tiny 4 层模型下把 device_map 压成 `{'': 0}` 单卡，**该测试目前未真正覆盖跨卡 forward**；M5 报告里需补一组 `max_memory` 强制切层的 case。

### 9.3 streaming kernel 段 1 `sparse_mode=4` 同款不靠谱（2026-05-25）

**触发**：`python -m pytest tests/test_streaming_kernel.py` 在 NPU 上 3 个 `test_npu_vs_pytorch_ref` 全 FAIL（max_abs_diff 1.28e-01 ~ 4.00e-01）；同时 `short_circuit_klen_lt_nlocal` 2 个 case nan FAIL（与下文无关，是测试用例 `s_q>s_k` 在 causal 下整行 -inf → softmax nan，已把用例改为 `s_q==s_k`）。

**根因**：`_streaming_npu` 段 1 原本走 `sparse_mode=4 + pre_tockens=n_local-1, next_tockens=0` 表达 sliding-window band，但 CANN 8.1.RC1 下 `sparse_mode=4` 与 §9.1 的 `sparse_mode=2` 同款 —— 不按文档语义生效，退化成 full attention，于是 sliding-window 被忽略，段 1 输出 ≠ 期望。

**修正**：段 1 改成 `sparse_mode=1 + 显式 [1,1,S_q,S_k] bool atten_mask`（`True=屏蔽`，与 M3/M4/段 2 一致），`try/except TypeError` 兜底 uint8。新增 `l1 = nan_to_num(l1, 0.0)` 与段 2 对齐，防御 LSE 合并的极端 nan。已修 `ops/streaming_kernel_npu.py::_streaming_npu` + 顶部 docstring。

**显存代价**：mask `[1,1,S_q,S_k]` 在 S=16384 内单测无压力；S>16384 已由 P7 silent dense fallback 兜底，本次不动。CANN 升 8.2+ 时可复测 `sparse_mode=4`，命中后回滚省 mask。

**CANN 8.2.RC1 复测结论（2026-05-25）**：sparse_mode=4 在 8.2.RC1 下仍退化为 full attention（与 §9.1 sm2 表同源），段 1 显式 mask 不能回滚。详见 §9.1 行为表。

**仍需盯**：M3 block-sparse / M4 vertical-slash 的 NPU 段已经全程显式 mask，理论上不踩此坑；M5 跑 `tests/test_block_sparse_kernel.py` + `tests/test_vertical_slash_kernel.py` 时确认。CANN 8.2.RC1 升级后 M5 全套 93 passed 已复核（2026-05-25）。

### 9.4 【v1 最大设计代价】minference per-head Python 循环 host-bound（2026-05-25）

**触发**：M5 第二轮 HF 端到端三组数据（Llama-3.1-8B fp16，单卡 npu:0）：

| ctx | attn | 时间 | 输出 | NPU AICore |
|---|---|---|---|---|
| 8K | minference | 308.53s | 正常 | ~0% |
| 16K | minference | 616.32s | 正常 | ~0% |
| 16K | dense | **4.57s** | **与 minference 字节级一致** | ~80% |

**根因**：`minference/modules/minference_forward.py:413-437` 对 32 个 query head 做 Python `for` 循环，每个 head 内部 ≥7 个 NPU 算子调用（QK 估计 / softmax / topk×2 / sort×2 / bool mask 构造 / `npu_fusion_attention`），prefill 一次的总算子调用数 ≈ 32 层 × 32 head × 7 ≈ **7000+ 次小 launch**。NPU 算完一个等 Python 派下一个 → device 持续空转。

对照：dense 路径每层 1 次 `npu_fusion_attention(H=32, S=16384)` 大算子，AICore 打满 80%。

**确认证据**：用户实机 `npu-smi info` 同时观察 dense（AICore 80%+）和 minference（持续 0%、偶尔 1%）；权重 24 GiB 在 HBM，排除 CPU 推理。

**v1 内不修**（架构层代价，立项即拍板"v1 不追性能"）。**v2 第一优先级**：消除 per-head 循环，三种可选路径：
1. batched 调度（首选）：32 head 的 v_idx/s_idx 一次性并行构造 + 单次 batched mask + 单次全 head FA
2. Triton-Ascend kernel（待 CANN 8.2+）
3. NPU 图模式 capture

**详细记录**：`docs/migration_v1_notes.md` §8；`docs/migration_v1_report.md` §5。

---

## 10. v1 验收（2026-05-25 收尾）

**结论**：v1 验收通过。算法 + 精度 + 链路全部交付，性能/长上下文上限作为已知设计代价精确量化。

| 维度 | 状态 |
|---|---|
| 算法迁移完整性 | ✅ 三类稀疏 kernel + dense + HF 链路全到位 |
| 精度 | ✅ 16K minference vs dense bit-identical |
| 性能 | ⚠️ minference 比 dense 慢 ~135×（host-bound），生产侧目前用 `--attn-type dense` |
| 长上下文 | ✅ 单卡 ≤32K；❌ 128K（bool mask O(S²) OOM） |
| 多卡 | ✅ accelerate 切层；❌ TP/SP（v2） |

**v2 启动前先解决两件事**（详见 `migration_v1_report.md` §9）：
1. 消除 per-head Python 循环（解锁性能）
2. CANN 升级到 8.2+ 复测 `sparse_mode=2/4`（可能直接省 O(S²) mask 显存）

**收尾产物**：`docs/migration_v1_notes.md`（13 条踩坑）+ `docs/migration_v1_report.md`（四维验收）+ 实测原始日志 `MInference-NPU/tests/M5_test_results{,_round2}.md`。

---

## 11. v2 启动计划（2026-05-25 锁定，等用户升完 CANN 回来执行）

### 11.1 环境升级（用户在 NPU 机执行，主对话不参与）

| 维度 | 旧（v1） | 新（v2） |
|---|---|---|
| 驱动 / npu-smi | 25.0.rc1.1 | **不变** |
| CANN Toolkit + Kernels | 8.1.RC1 | **8.2.RC1** |
| torch / torch_npu | 2.5.1 / 2.5.1 | **2.6.0 / 2.6.0** |
| Python | 3.10 | 不变 |
| transformers / accelerate / 其余 | 4.57.3 / 0.34.2 / ... | 不变 |

**为什么是 8.2.RC1（不是 8.3 也不是停在 8.1）**：是同时满足两个 v2 P0 目标（`sparse_mode=2/4` 修复 + 解锁 Triton-Ascend）的**最小升级**。8.3 跨度大、torch 要跳到 2.7/2.8、transformers 兼容性未验证；停在 8.1 等于没动。

**driver 25.0.rc1.1 不升**：用户明确不接 vllm-ascend 等下游推理框架，无需配套新驱动；CANN 8.2.RC1 与 25.0.rc1.1 驱动可兼容。

**已落地**：`MInference-NPU/requirements.txt` 的 torch / torch_npu 已改为 2.6.0，其余依赖未动。

**升级执行顺序**（用户侧 SOP，新 conda env 名 `minference-npu-cann82`，旧 env 保留作回滚）：
1. 备份旧 env：`conda env export > ~/minference-npu-cann81.yml`
2. 装新 CANN（toolkit + kernels-910b），并存 8.1.RC1 不覆盖
3. `source /usr/local/Ascend/ascend-toolkit/set_env.sh` → 校验 `ASCEND_HOME_PATH` 指向 8.2.RC1
4. `conda create -n minference-npu-cann82 python=3.10 -y && conda activate ...`
5. `pip install -r requirements.txt && pip install -e .`
6. 三层校验：`torch.npu.is_available()`、`python tests/test_env.py`、`pytest tests/`（全套 M1~M5 单测）

### 11.2 升完回来第一件事：复测 `sparse_mode` 行为表（**已执行 2026-05-25，分支 B 确认**）

复测脚本 `tests/test_sparse_mode_quirk.py`（B=1, N=4, S=256, D=128, fp16，对照手写 PyTorch eager causal/band），实测结果：

| 调用 | 期望（修复后） | 实测（CANN 8.2.RC1） | 影响 |
|---|---|---|---|
| `sparse_mode=2` 无 mask | causal | ❌ `max_abs=3.633e+00`，仍是 full | 路径 A mask 不能去 |
| `sparse_mode=4 + pre_tockens, next_tockens=0` | sliding-window band | ❌ `max_abs=3.633e+00`，与 full 几乎 bit-identical（vs full=2.441e-04）| streaming 段 1 mask 不能去 |
| `sparse_mode=1 + 显式 mask` | 保持正确 | ✅ `max_abs=1.953e-03` | 兜底路径正常 |

**结论：进入分支 B**。CANN 8.2.RC1 没修复 sm2/sm4，O(S²) bool mask 必须保留。
- 好消息：v1 现有显式 mask 路线在新环境零回归（M1~M5 全套 93 passed），算法层无任何变动需求
- 坏消息：v2 仍要靠分块 attention / 序列并行解决 128K OOM，工程量未减

**附带信息**：sm4 输出与 full attention 的 `max_abs=2.441e-04` 是新发现的确诊证据 —— 证明 `pre_tockens / next_tockens` 在 8.2.RC1 下被无视而非"按其他方式生效"，未来排查同类问题可作旁证。已同步 §9.1 / §9.3 行为表 + memory `project-minference-npu-fa-sparse-mode-quirk`。

### 11.3 v2 待解决问题清单（按优先级，详见两个 memory）

| 优先级 | 问题 | 解决路径（按推荐序） |
|---|---|---|
| **P0** | per-head Python 循环 host-bound（minference 比 dense 慢 ~135×） | (1) batched 32-head 调度 (2) Triton-Ascend kernel (3) `torch_npu.npu.graph_mode` |
| **P0** | O(S²) bool mask 显存（128K OOM） | ~~(1) sparse_mode=2/4 复测修复~~ **已证伪（§11.2，8.2.RC1 未修）** → (1) dense 分块 (2) Triton-Ascend 分块 (3) SP |
| P1 | 128K+ 多卡 attention（device_map=auto 仅切 layer） | TP / 序列并行 SP |
| P2 | minference 真稀疏路径 S>16384 silent fallback dense | 优化 topk/sort/`convert_vertical_slash_indexes` |

### 11.4 v2 启动前置 checklist（用户回来对一遍）

- [x] CANN 8.2.RC1 装好，`ASCEND_HOME_PATH` 指向新版本
- [x] 新 env `flexhead` 建好（注：实际命名而非计划的 `minference-npu-cann82`），`pip install -r requirements.txt` 无报错
- [x] `tests/test_env.py` PASS（实测 max_abs_diff=1.953e-03）
- [x] `pytest tests/` 全套 PASS（93 passed，M1~M5 在新环境零回归）
- [x] **`sparse_mode=2/4` 复测完成 — 分支 B 确认**，结果已回填 §9.1 / §9.3 / §11.2

**所有前置项已 done（2026-05-25）**。v2 实质开发按 §11.3 推进，但 O(S²) mask 行的"零开发选项"已划掉 —— 见 §11.2。

**关联 memory**：`project-minference-v1-perf-host-bound`、`project-minference-v1-mask-o-s2-ceiling`、`project-minference-npu-fa-sparse-mode-quirk`、`project-minference-v1-decisions`、`project-minference-ascend-migration`。


## 12. v2 PR-1 + PR-2 进度（2026-05-25 当晚）

### 12.1 已完成

**PR-1：消除 forward 顶层 per-head Python 循环**
- `minference/modules/minference_forward.py` —— `_vertical_and_slash_kernel` 入参支持 scalar 或长度 H 的 list/tensor（不同 head 用各自 vertical_size/slash_size 时走 V_max/S_max topk + duplicate-pad，保证 bit-identical）；forward 调度从 `for head in range(H)` 改为按 best_pattern 分组：vertical_and_slash heads 一次性 batched 调用，stream_llm / block_sparse 暂保留 per-head（占比 < 5%）
- `tests/test_minference_batched_vs.py` —— T1 同尺寸 batched ⇔ per-head；T2 不同尺寸 list 入参 ⇔ per-head；T3 scalar ⇔ 单元素 list 向后兼容回归

**PR-2：向量化 `_build_vs_mask_from_indexes`**
- `minference/ops/vertical_slash_kernel_npu.py` —— 旧实现重命名为 `_build_vs_mask_from_indexes_loop`（作为黄金参考保留），新增 `_build_vs_mask_from_indexes_vec`；模块级别名 `_build_vs_mask_from_indexes = _vec`，调用方 `_vertical_slash_npu` / `_vertical_slash_pytorch_ref` 自动切换
- 实现要点：用 `[B, H, num_rows, S_k]` 张量上的 `scatter_add_(±1) + cumsum > 0` 替代原 `for b: for h: for bq:` 三重 Python 循环；vertical 用 `int32 scatter_add + > 0` 避开 bool scatter 的覆写歧义；无效位用 sentinel=S_k 抵消，不污染 `[:S_k]` 切片
- 测试 T4 验证 `_loop ⇔ _vec` 在多组 (B, H, S, NNZ_V, NNZ_S) 下 `torch.equal == True`

**验证（NPU 服务器实测，env `flexhead`）**：
- T1/T2/T3/T4 全部 PASS（bit-identical 闭环）
- `pytest tests/ -v --ignore=tests/test_sparse_mode_quirk.py` 仍 93 passed，**零回归**

### 12.2 8K prefill 性能曲线

| 版本 | 8K + 4 decode 用时 | Δ vs v1 | 相对 dense | 收益来源 |
|---|---|---|---|---|
| v1（per-head + Python mask loop） | 308.53 s | — | 158× | — |
| v2 PR-1（batched VS） | 276.67 s | −32 s / 1.12× | 141× | 32 次 `npu_fusion_attention` launch 合并为 1 次 |
| v2 PR-2（vec mask build） | 139.30 s | −169 s / 2.2× | 71× | 4096 次/层的 host-bound 小 NPU launch → 几个大张量 op |
| **v2 PR-3（direct mask）** | **7.28 s** | **−301 s / 42×** | **3.73×** | 消除 `convert_vertical_slash_indexes` 双指针（~131072 次/prefill CPU iter）→ 整条 vs 路径无 Python 循环 |
| dense 8K baseline（同 NPU） | 1.95 s | — | 1.0× | 天花板：单 `npu_fusion_attention(H=32,S=8K)` 大算子 |

**输出文本与 dense 完全一致**（`' fox jumps over the'`），算法正确性确认。

`bit-identical` 维度：T1-T4 全程未破；T5/T6 验证 PR-3 算法变种（direct mask 是 v1 mask 的 visible 子集，差异仅来自上游 CUDA 块对齐扩展，详见 §12.4）。`pytest` 用户实测 PR-3 后全套 PASS（无回归）。

### 12.3 PR-2 当时的瓶颈分析（已由 PR-3 解决，仅留作历史参考）

**纯 NPU 计算量估算**：H=32, S=8K, D=128, fp16, 32 层 ≈ 8 TFLOPS，910B 上 ~25 ms 计算 + 几十 ms HBM 带宽 → **真实 NPU 时间预计 < 200 ms**，对比 PR-2 实测 139 s **意味着 >99% 时间消耗在 host 侧**。

**剩余瓶颈定位（PR-2 后）**：`minference/backend_npu/cuda_shim.py::convert_vertical_slash_indexes` 仍是纯 CPU Python 双指针 —— 每层调一次，内部 `for b: for h: for bq:` = 1 × 32 × 128 = **4096 次** `_process_block` + `torch.tensor(blk_buf, int32)` 分配 + `block_count[b,h,bq] = n` CPU 张量元素写，× 32 层 = **每次 prefill ~131072 次** Python + CPU PyTorch iter。

**PR-3 处理结果**：完全消除该路径 —— direct mask 在 NPU 上一步从 (v_idx, s_idx) 构建。8K prefill 从 139.30s → 7.28s（**42× 整体提速 / 19.1× 单步**），距离 dense 1.95s 天花板 3.73× —— host-bound 主导地位被打破，剩余开销主要在 NPU 端的 mask 构建张量算子（cumsum / scatter / repeat_interleave）+ 在线估计 topk。

### 12.4 PR-3：消除 convert_vertical_slash_indexes（2026-05-26 完成，代码已落地，待 NPU 实测）

**目标**：消除整条 v2 路径里**最后一处** host-bound CPU Python 循环 —— `convert_vertical_slash_indexes` 双指针每次 prefill ~131072 次迭代。

**实现**：`ops/vertical_slash_kernel_npu.py` 新增 `_build_vs_mask_direct(v_idx, s_idx, S_q, S_k, device, block_size)`，从 (v_idx, s_idx) 在 NPU 上一步生成 `[B,H,S_q,S_k]` bool mask，**完全跳过 convert**：
- Slash：对每个 (b,h,bq,k_s) 算 `range_end = max(end_m - s_raw, blk)`，invalid 位 sentinel 到 `S_k`，cumsum +1/-1 区间标记
- Vertical：v_idx scatter 一次得 `[B,H,S_k]`，broadcast 到所有 bq；因果 mask 兜底处理 `j >= end_m`
- `_vertical_slash_npu` 改走 direct（生产路径）
- `_vertical_slash_pytorch_ref` 也切到 direct（NPU 黄金参考）
- 保留 `_vertical_slash_pytorch_ref_legacy`（convert + loop，仅 T5/T6 对照用）

**算法差异（已确认非 bug）**：direct mask 与 v1 mask **非 bit-identical**。
- 上游 CUDA `vertical_slash_index.cu:69-72` 用**贪婪 blk 扩展**：相邻 slash 段让 `range_end += BLOCK_SIZE_M`（而非 `= new_range_end`），覆盖范围按 blk 累加。这是为 **Triton CUDA 块对齐 sparse kernel** 服务的近似。
- NPU 路径 A 用 token-level bool mask 喂 `npu_fusion_attention(sparse_mode=1)`，**无块对齐需求**，direct 用真实 slash 区间 OR 更精确、更稀疏。
- 关系：**direct True 集合 ⊇ v1 True 集合**（v1 多覆盖块对齐扩展位置；direct 可见 ⊆ v1 可见）。这是合法的 MInference 算法变种，**不破坏算法正确性**。
- 用户 2026-05-26 拍板：方案 B（用真实区间，放弃 bit-identical）。

**测试**（`tests/test_minference_batched_vs.py`，CPU 全套 16/16 PASS）：
- T1/T2/T3：batched 等价（_pytorch_ref 同走 direct，仍 bit-identical 闭环）
- T4：loop vs vec mask 构建（PR-2 遗留，与 PR-3 无关）
- **T5（新）**：direct mask True ⊇ v1 mask True 子集关系（7 组参数，验证安全约束）
- **T6（新）**：mask 一致行 attention 输出 bit-identical（隔离差异来源 —— S=512/384 case 下 mask 完全一致 max_abs=0，S=256 case 392/512 一致行内 bit-close，证明差异仅来自 v1 块对齐扩展，不是实现 bug）

**修改文件清单**：
- `minference/ops/vertical_slash_kernel_npu.py`（新增 `_build_vs_mask_direct` + `_vertical_slash_pytorch_ref_legacy`；`_vertical_slash_npu` / `_pytorch_ref` 切到 direct）
- `tests/test_minference_batched_vs.py`（新增 T5/T6 + __main__ 入口更新）

**实测验证（2026-05-26 用户完成）**：
- [x] NPU 服务器（env `flexhead`）`pytest tests/ -v --ignore=tests/test_sparse_mode_quirk.py` 全套 PASS，零回归 + T5/T6 新增 PASS
- [x] `examples/run_hf_minimal.py --attn-type minference --ctx-len 8192 --max-new-tokens 4`：**7.28s**（对比 PR-2 139.30s，单步 19.1× 提速 / 整体 42× from v1）
- [x] dense 8K baseline：1.95s（天花板）—— PR-3 相对 dense **3.73× slow**，host-bound 主导地位已破
- [x] 输出文本与 dense 一致 `' fox jumps over the'`，算法正确性确认

性能表已回填到 §12.2。

### 12.5 PR-3 之后路径 A 的终结（2026-05-26 用户实测 16K + 拍板弃用）

**16K 实测**（同 dense 8K 命令换 ctx-len=16384）：

| ctx | attn | 时间 | 相对 dense |
|---|---|---|---|
| 8K | minference | 7.28s | 3.73× |
| 8K | dense | 1.95s | 1.0× |
| 16K | minference | **21.95 s** | **4.69×** |
| 16K | dense | 4.68 s | 1.0× |

**问题暴露**：差距从 8K 的 3.73× **扩大到 16K 的 4.69×**，不收敛。原因如 §13.1 §13.2 阐述 —— **路径 A 不是真稀疏，永远跑不过 dense**。

---

## 13. v2 路径 B：triton-ascend 真稀疏 FA kernel（PR-4，2026-05-26 立项）

### 13.1 为什么路径 A 是死胡同

**核心结论**：`npu_fusion_attention + bool mask` 不是真稀疏 —— mask 只把部分位置 softmax 设 -inf，**计算量仍是 H × S² × D 的 dense QK matmul**，没有 FLOPs 节省。

| 维度 | 路径 A 现状 | 影响 |
|---|---|---|
| 算法本质 | dense FA + mask 屏蔽 | **永远比 dense 慢**（多了 mask 构建开销） |
| O(S²) bool mask 显存 | 8K=64MiB / 16K=256MiB / 32K=1GiB / 128K=16GiB | **128K 必 OOM**，32K 是单卡甜区 |
| MInference 算法价值 | 仅保留 "按 v_idx/s_idx 选 token" 骨架，丢失 "少算 K/V" 的本质收益 | 项目核心价值丧失 |
| 路径 A 优化天花板 | 逼近 dense（永远到不了） | 继续优化无意义 |

**继续在路径 A 上优化最多只能减少 host-bound 开销，无法解决根本问题**。PR-3 已经把 host-bound 主导消除（19.1× 单步提速），但仍输 dense 3.73-4.69×，且差距随 S 扩大，**确证路径 A 是设计死胡同**。

### 13.2 路径 B 立项：triton-ascend 真稀疏 FA kernel

参考上游 MInference 的 Triton sparse FA kernel（`MInference/minference/ops/pit_sparse_flash_attention_v2.py` 等），按 v_idx/s_idx 的 block **跳过整块 K/V 计算**，**不构建 mask、计算量真减少**。

NPU 对应方案：**triton-ascend**（华为开源 Triton for Ascend）。

**为什么 v1 当时排除 triton-ascend**：CANN 8.1.RC1 与 triton-ascend 版本矩阵不匹配（§2 决策）。**现在升级到 CANN 8.5.0 + triton-ascend 3.2.0 后该限制解除**。

**外部参考**：
- triton-ascend 3.2.0：https://github.com/triton-lang/triton-ascend
- tilelang-ascend（更高层 DSL，已有 SparseFlashAttention / DeepSeek V4 reference 实现）：https://github.com/tile-ai/tilelang-ascend
- 上游 MInference Triton VS kernel：`MInference/minference/ops/pit_sparse_flash_attention_v2.py`

### 13.3 环境升级目标（用户 2026-05-26 选定 B 选项）

| 维度 | 旧（v1/v2 路径 A） | 新（v2 路径 B / PR-4） |
|---|---|---|
| 驱动 / npu-smi | 25.0.rc1.1 | **不变**（CANN 8.5.0 与 25.0.rc1.1 兼容） |
| CANN Toolkit + Kernels | 8.2.RC1 | **8.5.0** |
| torch / torch_npu | 2.6.0 / 2.6.0 | **2.7.1 / 2.7.1.post4** |
| triton-ascend | 不装 | **3.2.0**（pip install triton-ascend，注意 community Triton 互斥） |
| Python | 3.10 | 不变（3.9-3.11 都行） |
| transformers / accelerate | 4.57.3 / 0.34.2 | 暂不变，升完 CANN 后验证兼容；如挂再升 |

**官方矩阵**（2026-05 调研结果）：
| triton-ascend | CANN | torch_npu | Python | release |
|---|---|---|---|---|
| 3.2.1 | 9.0.0 | 2.7.1.post4 | 3.9-3.11 | 2026-04-30（跨度大，不取） |
| **3.2.0** | **8.5.0** | **2.7.1.post4** | **3.9-3.11** | **2026-01-16（用户选定）** |
| 3.2.0rc4 | 8.3.RC2 / 8.3.RC1 | 2.7.1.post4 | 3.9-3.11 | 2025-11-20（rc 不取） |

**用户侧升级 SOP**（新 conda env 名建议 `flexhead-c85`，旧 `flexhead` 保留作回滚）：
1. 备份旧 env：`conda env export -n flexhead > ~/flexhead-cann82.yml`
2. 装新 CANN 8.5.0（toolkit + kernels-910b），不覆盖 8.2.RC1
3. `source /usr/local/Ascend/ascend-toolkit/set_env.sh` → 校验 `ASCEND_HOME_PATH` 指向 8.5.0
4. `conda create -n flexhead-c85 python=3.10 -y && conda activate flexhead-c85`
5. `pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu`
6. `pip install torch_npu==2.7.1.post4`
7. `pip install triton-ascend==3.2.0 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi`（注意：装 triton-ascend 前确认无 community Triton 占位）
8. 把 `MInference-NPU/requirements.txt` 更新到 2.7.1，`pip install -r requirements.txt && pip install -e .`
9. 三层校验：
   - `python -c "import torch; import torch_npu; import triton_ascend; print(torch.npu.is_available())"`
   - `python tests/test_env.py`
   - `pytest tests/ -v --ignore=tests/test_sparse_mode_quirk.py`（M1-M5 复测，理论 93+ passed）
10. 复测 `sparse_mode=2/4`（CANN 8.5.0 可能修复，详见 §11.2 / §13.5）

### 13.4 PR-4 路线图

| PR | 内容 | 工作量 | 主要风险 |
|---|---|---|---|
| **PR-4-env** | 用户侧环境升级（见 §13.3） | 0.5 天 | M1-M5 在新环境的回归（PR-3 mask 路径 A 应该都仍跑通，因为不依赖 triton-ascend） |
| **PR-4-poc** | triton-ascend 最简 dense FA kernel + 与 `npu_fusion_attention` dense 8K/16K 对比 | 3-5 天 | triton-ascend 对 attention 类 kernel 的成熟度；性能是否同量级 |
| **PR-4-archive** | path A 代码搬到 `minference/legacy/path_a/`，入口简化 | 1-2 天 | 测试要保留作 deprecated 回归 |
| **PR-4-VS** | triton-ascend 真稀疏 vertical-slash FA kernel（核心 PR） | 2-3 周 | 复杂控制流（按 v_idx 跳块）；block size 调优；NPU 内存层次适配 |
| PR-4-BS | triton-ascend block-sparse FA kernel | 1-2 周 | 同上 |
| PR-4-SL | triton-ascend streaming kernel | 1-2 周 | 同上 |
| **PR-4-bench** | 8K / 16K / 32K / 64K / **128K** 长上下文实测 + 性能/精度报告 | 1 周 | 32K+ 的显存 / 调度问题 |

**总工作量**：6-10 周（除环境升级 0.5 天）

### 13.5 PR-4 验收标准

| 维度 | 目标 |
|---|---|
| 算法 | 真稀疏：minference vertical-slash 算出与 dense **容差对比**（fp16 max_abs < 1e-2） |
| 性能 8K | minference < dense（路径 A 永远到不了） |
| 性能 32K | minference 显著 < dense（稀疏度提升放大优势） |
| 长上下文 | **128K 单卡可跑**（路径 A 必 OOM 的红线） |
| 测试 | 全套 M1-M5 + PR-4 新增（kernel 单测 + 长上下文端到端） |
| 显存 | 128K 单卡 fp16 估算 ≤ 64 GiB（910B3 单卡 64GiB HBM，刚好可跑） |

### 13.6 路径 A 历史价值（虽然弃用，但仍有意义）

- 验证了 `npu_fusion_attention` + bool mask 的完整链路（patch → forward → kernel → mask）
- 验证了 NPU 上 transformers 4.57 / accelerate device_map 等基础设施
- 提供了 `_vertical_slash_pytorch_ref` 等 CPU reference 实现，可继续作为 PR-4 kernel 的黄金参考
- M5 实测产出的 host-bound profile 数据 + PR-1/PR-2/PR-3 的优化经验，对 PR-4 kernel 调优有借鉴价值

→ **不删除路径 A 代码，搬到 legacy/ 归档**。

### 13.7 PR-4-tl-sfa 闸门（tilelang-ascend 官方 SFA 接口探针，2026-05-26）

**目标**：在正式重写 A-shape / block-sparse kernel 前，先确认 tilelang-ascend 官方 `examples/sparse_flash_attention/example_sparse_flash_attn.py::sparse_attention_fwd` 的真实接口、输入顺序、indices 语义和精度上限。

**实测环境**：NPU 服务器 env `flexhead-tl`，`PYTHONPATH=~/tilelang-ascend`。导入官方 example 会先跑其 top-level smoke test，日志里会出现 `init successful!` / `Test Passed!`，这是预期行为。

**真实接口（已 probe）**：
- 构造器签名：`sparse_attention_fwd(heads, dim, tail_dim, topk, kv_stride, kv_group=1, sm_scale=None, is_causal=True, block_I=64)`
- kernel 调用：`kernel(q, kv, indices)` 三参数；`Output` 和 5 个 workspace 由 `@tilelang.jit(out_idx=[3], workspace_idx=[4,5,6,7,8])` 自动分配。
- `q`: BSHD `[B, S_q, H, dim + tail_dim]`
- `kv`: BSGD `[B, S_k, kv_group, dim + tail_dim]`，NSA/DeepSeek 风格 packed KV；reference 里 `k = kv`，`v = kv[..., :dim]`
- `indices`: `[B, S_q, kv_group, topk]` int32，每项是 K token 位置，不是 block id
- `out`: `[B, S_q, H, dim]`
- 默认 `sm_scale = (dim + tail_dim) ** -0.5`
- causal 参考的 `q_start_index_s` 默认是 `S_k * kv_stride - S_q`，即 Q 是 KV 尾部窗口。

**当前已跑通的小尺寸闸门参数**：
- `B=1, S_q=128, S_k=512, H=4, kv_group=1, dim=128, tail_dim=128, topk=256, block_I=64, kv_stride=1, q_start=384`
- `KV_GROUP=1` 是官方 example 硬断言；当前不能直接表达 per-head MHA indices 或 GQA。
- `topk` 列必须全部是有效 K token；实测 `pad=S_k` 会导致 tilelang kernel 输出 NaN。

**已通过测试**（2026-05-26 用户实测）：
| case | 命令 | 结果 |
|---|---|---|
| sanity | `PYTHONPATH=~/tilelang-ascend python tests/test_tilelang_sfa_integration.py --case sanity` | PASS，`max_abs_diff=4.8828e-04`, `mean_abs_diff=1.4170e-05` |
| block-sparse | `PYTHONPATH=~/tilelang-ascend python tests/test_tilelang_sfa_integration.py --case bs` | PASS，`max_abs_diff=2.4414e-04`, `mean_abs_diff=1.4100e-05` |
| stream-llm / A-shape | `PYTHONPATH=~/tilelang-ascend python tests/test_tilelang_sfa_integration.py --case sl` | PASS，`max_abs_diff=2.4414e-04`, `mean_abs_diff=1.4100e-05` |

**代码状态**：
- `MInference-NPU/tests/test_tilelang_sfa_integration.py` 已改成真实 `q, kv, indices` 三参数路径，并内置官方 reference 的参数化版本；该脚本是手跑 NPU 闸门，不作为 pytest collection 目标。
- `MInference-NPU/minference/ops/tilelang_indices.py` 已有 `block_indices_to_tilelang` / `stream_llm_to_tilelang`，其中 stream-llm 对 anchor/local 重叠的 local 项填 `pad_value`，避免真实 sparse kernel 重复计入同一个 K token。
- `MInference-NPU/tests/test_tilelang_indices.py` 已同步 stream-llm 重叠行为预期；但在 `flexhead-tl` 环境跑 `python -m pytest tests/test_tilelang_indices.py -v` 会因该 env 没装 `transformers` 而 collection 失败，因为测试直接 `from minference.ops...` 会触发 `minference/__init__.py` eager import。后续有两种处理：在 `flexhead-tl` 装 `transformers`，或把该测试改成像 `test_tilelang_sfa_integration.py` 一样按文件路径 standalone 加载 `tilelang_indices.py`。

**与最终目标的差距（非常重要）**：
1. 你当前只需要 **A-shape / stream-llm** 和 **block-sparse** 两种模式；tilelang 官方 SFA 是 NSA packed-KV kernel 原型，不是 MInference 标准 K/V 分离 attention 的 drop-in 替代。
2. K/V 语义不同：MInference 标准输入是分离的 `q, k, v`；官方 kernel 是 packed `kv`，且 `v = kv[..., :dim]`。
3. `kv_group==1` 限制：当前官方 example 只支持所有 Q heads 共享一组 KV/Indices，不能直接表达 per-head sparse pattern。
4. pad 不容错：真实 A-shape / block-sparse 早期 token 或不足 topk 时天然有无效槽位，官方 kernel 目前遇到 `pad=S_k` 会 NaN。
5. 因此下一步不是“直接替换”，而是先设计适配层：标准 K/V → packed KV、per-head pattern → shared/group indices、pad/不足 topk 的处理策略。

**建议下一步顺序**：
1. 先修 `tests/test_tilelang_indices.py` 的 standalone import，保证 `flexhead-tl` 无 transformers 也能跑 CPU-only indices 单测。
2. 保持 PR-4 范围收窄：只做 A-shape 和 block-sparse，不做 vertical-slash。
3. 优先 block-sparse：更容易构造满 `topk` 的有效 K block，先做共享 pattern MVP。
4. 再做 A-shape：需要处理 early token、anchor/local 重叠和不足 topk 的策略。
5. 决定 `kv_group==1` 的短期策略：共享 pattern MVP / per-group union / 修改 tilelang kernel 支持 `kv_group>1`。
6. 决定 packed KV 策略：若要保持普通 attention 精确语义，官方 `v = kv[..., :dim]` 路径不够，需要改 kernel 支持独立 K/V，或另写更贴近 MInference 的 tilelang/triton kernel。

---

## 14. 当前下一步（PR-4-tl-sfa 之后继续）

- [x] NPU 服务器 `flexhead-tl` 已可导入 tilelang-ascend 官方 SFA，官方 smoke test PASS。
- [x] PR-4-tl-sfa 三个闸门 case（sanity / block-sparse / stream-llm）均 PASS。
- [ ] 修 `tests/test_tilelang_indices.py` 在 `flexhead-tl` 无 transformers 环境下的 import 问题。
- [ ] 继续 PR-4 适配设计：只覆盖 A-shape 和 block-sparse，先不做 vertical-slash。
- [ ] 明确三项关键设计：pad 处理、`kv_group==1`/per-head pattern 处理、标准 K/V 到 packed KV 的处理。
