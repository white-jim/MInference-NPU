# 上下文检查点 — MInference 1.0 → 昇腾 NPU 算法迁移

> 用途：在新会话中快速恢复工作上下文。详细信息在各专题文档里，本文件只列必读关键点。
> 最近更新：2026-05-25 — M5：`test_env` + `test_dense_forward` + `test_streaming_kernel`（32/32，含 3 NPU）+ `test_block_sparse_kernel`（29/29，含 3 NPU）+ `test_vertical_slash_kernel`（27/27，含 1 NPU）实机 ALL PASS；修复 transformers 4.57.3 attention 新签名兼容（§9.2）；streaming kernel 段 1 `sparse_mode=4`→`sparse_mode=1+显式 mask`（§9.3）。剩 HF 端到端 `examples/run_hf_minimal.py`。
> 工作目录：`D:\works\算法迁移\`（不是 git repo；子目录 `MInference-NPU/` 是 git repo）

---

## 1. 任务一句话

把 Microsoft [MInference 1.0](https://github.com/microsoft/MInference) 长上下文 LLM 推理加速算法迁移到 **华为昇腾 NPU**，v1 走 **`npu_fusion_attention` + bool mask（路径 A）** 一条路（Triton-Ascend 留 v2，原因见 `MInference-NPU/docs/SETUP.md` §2：CANN 8.1.RC1 与 triton-ascend 版本矩阵不匹配），宿主 **HF transformers + torch_npu**。

- 上游源码（参考蓝本，**不动**）：`MInference/`
- v1 产出主线：`MInference-NPU/`
- 全部专题文档：`docs/`

---

## 2. v1 决策（已拍板，写在 `docs/migration_plan_v1.md`）

| 维度 | 选择 |
|---|---|
| 迁移范围 | vertical_and_slash + block_sparse + stream_llm + dense fallback |
| 排除项 | dilated / static / tri_shape / kvcompress / dist_ops / vLLM 集成 / FA3 / KV-CPU offload |
| 算子主路径 | `npu_fusion_attention` + bool mask（路径 A）；Triton-Ascend 留 v2（CANN 8.1.RC1 下版本矩阵不匹配） |
| `convert_vertical_slash_indexes` | CPU Python 双指针（M4-a）；NPU kernel 留 v2 |
| 框架宿主 | HF transformers + torch_npu；accelerate `device_map="auto"` 多卡 OK |
| 产出形态 | 独立 `MInference-NPU/` 完整 py 源 + tests + examples + docs + 独立 setup.py |

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
[~] M5 端到端联调 + 精度/性能/效果报告（test_env + test_dense_forward + test_streaming_kernel + test_block_sparse_kernel + test_vertical_slash_kernel PASS；HF 端到端 examples/run_hf_minimal.py 待跑）
[ ] docs/migration_v1_notes.md + docs/migration_v1_report.md
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

### 9.1 `npu_fusion_attention` 的 `sparse_mode` 在 CANN 8.1.RC1 下不靠谱（2026-05-25）

**触发**：`python tests/test_env.py -v` 中 `test_npu_fusion_attention_smoke` FAIL，`max_abs_diff=3.633e+00`，差异跟"causal 期望 vs full attention 实测"对得上。

**实测组合**（B=1, N=4, S=256, D=128, fp16，与手写 PyTorch eager causal 比对）：

| 调用 | `max_abs_diff` | 实际语义 |
|---|---|---|
| `sparse_mode=0`（无 mask） | ~3.6 | full attention |
| `sparse_mode=2`（无 mask） | ~3.6 | full attention（**不是** causal） |
| `sparse_mode=2 + pre_tockens/next_tockens` | ~3.6 | full attention |
| `sparse_mode=1 + 显式 [S_q, S_k] bool atten_mask` | ~0.00195（mean ~2e-5） | 正确 causal |

**修正**：路径 A 全部 causal 调用改走 `sparse_mode=1 + 显式 bool atten_mask`（`True=masked`，与 M3/M4 既有约定一致），`try/except TypeError` 兜底 uint8。已修 `backend_npu/attention.py::dense_attention` + `tests/test_env.py` + `docs/SETUP.md` §5/§5.1。修正后实机 smoke max_abs_diff=1.953e-03，PASS。

**仍需盯**：`streaming_kernel_npu.py` 的 sliding-window 段用 `sparse_mode=4`，未单独实测过同类问题，跑 `tests/test_streaming_kernel.py` 时复核精度。CANN 升到 8.2+ 时可复测 `sparse_mode=2` 决定是否回滚省 mask 显存。

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

**仍需盯**：M3 block-sparse / M4 vertical-slash 的 NPU 段已经全程显式 mask，理论上不踩此坑；M5 跑 `tests/test_block_sparse_kernel.py` + `tests/test_vertical_slash_kernel.py` 时确认。
