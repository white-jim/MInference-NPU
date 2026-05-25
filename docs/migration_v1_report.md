# MInference 1.0 → 昇腾 NPU 迁移验收报告 v1

> 用途：对 v1 的最终交付物做**功能、精度、性能、长上下文能力**四维实测对比，给出结论与 v2 启动建议。
> 创建日期：2026-05-25
> 关联文档：`docs/context_checkpoint.md`、`docs/migration_v1_notes.md`、`docs/migration_plan_v1.md`、`MInference-NPU/docs/{SETUP,M1_dense_pipeline,M2_streaming,M3_block_sparse,M4_vertical_slash}.md`
> 实机测试数据原始日志：`MInference-NPU/tests/M5_test_results.md` + `MInference-NPU/tests/M5_test_results_round2.md`

---

## 0. 一句话结论

**v1 完成"算法 + 精度迁移"，但"性能加速"未兑现，原因是 v1 的设计代价（per-head Python 循环 host-bound + 显式 bool mask O(S²) 显存）在实机上被精确量化。** 该结论与立项决策一致（`migration_plan_v1.md` §0："v1 只追算法跑通 + 精度对齐，不追性能"），不构成 v1 验收失败，但 **v2 启动前必须先解决 per-head 循环和 mask 显存这两个根因**，否则 minference 路径在生产侧没有意义。

---

## 1. 测试环境

| 项 | 内容 |
|---|---|
| 硬件 | 华为昇腾 910B3 ×4（每卡 60 GiB HBM），实际使用 NPU 0~3 |
| OS | Linux aarch64 |
| CANN | 8.1.RC1 |
| torch_npu | 2.5.1 |
| Python | 3.10.20 |
| transformers | 4.57.3 |
| 测试模型 | meta-llama/Llama-3.1-8B-Instruct（fp16 加载） |
| 模型存放 | `/data/guoshiyao/resources/models/Meta-Llama-3.1-8B-Instruct`（标准 HF 本地目录） |
| 模型结构 | LlamaForCausalLM，32 层，hidden=4096，Q heads=32，KV heads=8（GQA），head_dim=128，max_pos=131072 |

---

## 2. 功能验收：迁移范围实现完整性

### 2.1 v1 范围

| 模块 | 上游来源 | v1 实现位置 | 状态 |
|---|---|---|---|
| 顶层入口 `MInference` 类 | `models_patch.py` | `MInference-NPU/minference/models_patch.py` | ✅ |
| HF monkey-patch | `patch.py` | `MInference-NPU/minference/patch.py` | ✅ |
| Per-head 调度 + LlamaAttention forward 改写 | `modules/minference_forward.py` | `MInference-NPU/minference/modules/minference_forward.py` | ✅（transformers 4.57.3 签名适配） |
| NPU dense 算子薄封装 | flash_attn / SDPA | `MInference-NPU/minference/backend_npu/attention.py`（`npu_fusion_attention + bool mask`） | ✅ |
| `convert_vertical_slash_indexes` CUDA → Python 双指针 | `csrc/vertical_slash_index.cu` | `MInference-NPU/minference/backend_npu/cuda_shim.py` | ✅（M4-a） |
| Streaming kernel | `ops/streaming_kernel.py` Triton | `MInference-NPU/minference/ops/streaming_kernel_npu.py` | ✅（M2） |
| Block-Sparse kernel | `ops/block_sparse_attention.py` Triton | `MInference-NPU/minference/ops/block_sparse_kernel_npu.py` | ✅（M3） |
| Vertical-Slash kernel | `ops/vertical_slash_attention.py` Triton | `MInference-NPU/minference/ops/vertical_slash_kernel_npu.py` | ✅（M4-b） |
| 三类 attn_type | `attn_type ∈ {minference, dense, hf}` | 同左，参数维度 v1 仅保留 3 种 | ✅ |

### 2.2 v1 明确**未做**项（v2 计划）

dilated / static / tri_shape / kvcompress / dist_ops / vLLM 集成 / FA3 / KV-CPU offload / Triton-Ascend 主路径 / `convert_vertical_slash_indexes` NPU 化。详见 `migration_plan_v1.md` §0。

---

## 3. 单元测试验收：全部 PASS

实机执行 `python -m pytest tests/test_*.py -v`，按 kernel 维度统计：

| 测试文件 | 用例数 | 含 NPU case | 结果 |
|---|---|---|---|
| `tests/test_env.py` | 4 | 1（npu_fusion_attention smoke） | ✅ ALL PASS |
| `tests/test_dense_forward.py` | 5 | 全 NPU | ✅ ALL PASS |
| `tests/test_streaming_kernel.py` | 32 | 3 | ✅ 32/32 PASS |
| `tests/test_block_sparse_kernel.py` | 29 | 3 | ✅ 29/29 PASS |
| `tests/test_vertical_slash_kernel.py` | 27 | 1 | ✅ 27/27 PASS |

**精度上限（kernel 层）**：

| Kernel | dtype | vs eager / pytorch ref `max_abs_diff` |
|---|---|---|
| dense (`npu_fusion_attention`+mask) | fp16 | 1.953e-03 |
| Streaming | fp16 | ≤ 0.4（M5 调整 sparse_mode 后所有 NPU case 通过容差） |
| Block-Sparse | fp16 | 容差内 |
| Vertical-Slash | fp16 | 容差内 |

具体踩坑记录（含 sparse_mode=2/4 退化 bug 的发现过程）见 `migration_v1_notes.md` §2。

---

## 4. 精度验收：HF 端到端 minference vs dense bit-identical

测试方法：用同一 prompt（`"The quick brown fox jumps over the lazy dog. "` 循环拼到指定 ctx 长度），同一 model、同一 seed（greedy `do_sample=False`），切换 `--attn-type` 在 `minference / dense` 之间对比 16 token greedy 输出。

| ctx | attn_type | 输出文本 | 与 dense 一致？ |
|---|---|---|---|
| 16 K | minference | `' lazy dog. The quick brown fox jumps over the lazy dog. The quick brown'` | — |
| 16 K | dense | `' lazy dog. The quick brown fox jumps over the lazy dog. The quick brown'` | ✅ 完全一致 |
| 32 K | minference | `' over the lazy dog. The quick brown fox jumps over the lazy dog. The'` | — |
| 32 K | dense | `' over the lazy dog. The quick brown fox jumps over the lazy dog. The'` | ✅ 完全一致 |

**结论**：v1 minference 的 per-head 稀疏估计 + 三类 kernel 在端到端 greedy 解码上对 dense 基线**字节级等价**。

**注意**：32 K 的 minference 实际触发了 vertical-slash silent dense fallback（见 `migration_v1_notes.md` §9），所以 32K 这一行本质是"以稀疏调度方式跑 dense"，等价是平凡的；**16 K 才是真稀疏路径与 dense 的真正等价验证**。

---

## 5. 性能验收：minference 比 dense 慢 ~135 倍，host-bound

### 5.1 端到端延迟实测

prefill + 16 token greedy decode，单卡 NPU 0：

| ctx | attn_type | 总时间 (s) | 备注 |
|---|---|---|---|
| 8 K | minference | 308.53 | 稳态（复跑 308 / 309 两次一致，排除 JIT 编译嫌疑） |
| 16 K | minference | 616.32 | 与 8 K 几乎线性翻倍 |
| 16 K | dense | **4.57** | minference 慢 **134.9×** |
| 32 K | minference | 12.17 | 已触发 silent dense fallback |
| 32 K | dense | 10.54 | dense 直跑 |
| 32 K | hf（无 patch） | OOM | eager attention `[B,H,32K,32K]` 申请 64 GiB，超 60 GiB 单卡 |

### 5.2 NPU AICore 利用率实测

跑 16 K 时 `npu-smi info` 实时观察：

| attn_type | AICore 利用率 | HBM | 解读 |
|---|---|---|---|
| dense | **80% 左右**（持续高位） | ~20 GiB | NPU 算力被打满 |
| minference | **0%，偶尔窜到 1%** | ~20 GiB | NPU 等 host，算力空转 |

权重确实在 HBM（24 GiB ≈ 8B fp16 + KV + 激活），完全排除"退化到 CPU 推理"的可能性。

### 5.3 根因：per-head Python 循环

定位到 `MInference-NPU/minference/modules/minference_forward.py:413-437`：dense 路径每层一次 `npu_fusion_attention(B=1, H=32, S=16384)` 直接打满 AICore；minference 路径对 32 个 query head 做 Python `for` 循环，每个 head 内部还要做 QK 估计 / softmax / topk×2 / sort×2 / bool mask 构造 / 再调 `npu_fusion_attention(B=1, H=1, S=16384)`，单 head 单次大概 7+ 个小算子。

prefill 总算子调用数对比：

| 路径 | 大算子调用数 | 单调用规模 |
|---|---|---|
| dense | 32 次 | B=1, **H=32**, S=16384 |
| minference | 32 层 × 32 head × 7+ ≈ **7000+** 次 | B=1, **H=1**, S=16384 |

NPU 的 host launch overhead 比 GPU 大一个量级以上（典型经验值），7000+ 次小 launch 把 host 端 Python 完全堆成瓶颈。详细分析与 v2 解法见 `migration_v1_notes.md` §8。

### 5.4 性能验收结论

v1 minference **没带来加速，反而比 dense 慢 ~135 倍**。这与立项决策一致（不追性能），但作为正式验收数据必须如实记录：

- **生产可用的路径目前是 dense**（`--attn-type dense`），它正是 v1 在 NPU 上提供的"`npu_fusion_attention + bool mask` 基线"
- **minference 路径在 v1 阶段仅用作算法正确性验证基线**，性能不可用
- v2 第一优先级是消除 per-head Python 循环（见 §8）

---

## 6. 长上下文能力验收：v1 单卡甜区 ≤ 32K

### 6.1 实测上限矩阵（单卡 910B3 60 GiB）

| ctx | attn_type | 状态 | 限制因素 |
|---|---|---|---|
| 8 K | minference / dense / hf | ✅ 通 | 无 |
| 16 K | minference / dense | ✅ 通 | 无；hf 未单测但理论 OK（attn_weights 1 GiB） |
| 32 K | minference (实为 dense fallback) / dense | ✅ 通 | hf OOM（attn_weights 64 GiB） |
| 64 K | （未测） | ⚠️ 边界 | dense bool mask ≈ 4 GiB，理论可跑 |
| 128 K | minference 4 卡 | ❌ OOM | dense fallback 的 `causal_mask` 单张 17 GiB |

### 6.2 多卡（accelerate `device_map="auto"`）的实际作用

128 K 4 卡 minference 实测 NPU 0 = 58 GiB（满）、NPU 1~3 各约 4 GiB（仅权重副本）。accelerate 按 layer 边界切模型，但**单层 attention 的全部中间激活（QK^T、bool mask、softmax、output）都必须在该层所在卡上分配**。所以多卡只摊薄了权重，无法缓解 attention 内部的瞬时显存峰值。

要把超长 ctx 跑起来，必须解决**单卡内 attention 计算的显存峰值**，即 §3 / §10 (`migration_v1_notes.md`) 的两个根因，不能寄希望于堆卡。

### 6.3 长上下文能力结论

| 场景 | v1 是否覆盖 |
|---|---|
| ≤ 16 K 单卡 prefill + decode | ✅ |
| 16 K ~ 32 K 单卡 prefill + decode | ✅（minference 触发 silent dense fallback） |
| 32 K ~ 64 K 单卡 | ⚠️ 理论可跑，未实测，建议补 |
| 128 K 单卡 / 多卡 | ❌ v1 不支持 |

---

## 7. v1 vs 上游 GPU 版差距盘点

| 维度 | 上游 GPU (CUDA + Triton) | v1 NPU | 差距来源 |
|---|---|---|---|
| 算法范围 | 全（11 种 attn_type + 8 种 kv_type） | 3 种 attn × 1 种 kv | v1 立项即缩减 |
| 精度 | 浮点合理误差 | 与 dense bit-identical（16K 验证） | 无差距 |
| Prefill 性能（128K Llama-8B） | 加速明显（上游 paper 报告 10×+） | minference 比 dense 慢 ~135× | **per-head Python 循环 + 缺 fused kernel** |
| 显存能力 | FA-style 分块，不物化 attn_weights | bool mask 物化 O(S²) | CANN 8.1.RC1 `sparse_mode=2/4` 失效，被迫显式 mask |
| 长上下文上限 | 128K~1M | v1 单卡 ≤32K | 上一行的直接后果 |
| 多卡 | TP / SP / PP | 仅 accelerate PP | v1 立项即缩减 |

---

## 8. v1 验收结论

按立项目标（`migration_plan_v1.md` §0 "v1 只追算法跑通 + 精度对齐"）：

| 立项目标 | 验收结果 |
|---|---|
| MInference 三类稀疏算法迁移到 NPU 跑通 | ✅ 完成 |
| HF transformers + torch_npu 宿主链路打通 | ✅ 完成 |
| accelerate 多卡部署可用 | ✅ 完成（权重切分） |
| 精度对齐 dense 基线 | ✅ bit-identical |
| 不引入与 v2 路径对立的写法 | ✅（per-head 循环可在 v2 替换为 batched 调度；bool mask 可在 CANN 升级后切到 sparse_mode=2） |

**v1 验收通过**。性能与长上下文能力的限制是立项时已知的设计代价，本次实机把代价**精确量化**。

---

## 9. v2 启动建议

按重要性 + 投入回报排序：

### 9.1 必做（解锁性能）

1. **消除 per-head Python 循环**（`minference_forward.py:413-437`）。建议优先方案：把 32 个 head 的 v_idx / s_idx 一次性 batched 构造，单次 batched bool mask + 单次全 head `npu_fusion_attention`。预计能把 16 K minference 从 616 s 拉到与 dense 同量级（个位数秒）。
2. **CANN 升级到 8.2+**，复测 `sparse_mode=2/4` 是否修复。如修复，dense 路径可省掉 O(S²) bool mask，把单卡 ctx 上限从 32 K 推到 128 K+。

### 9.2 解锁更长 ctx

3. **`convert_vertical_slash_indexes` NPU 化**（M4-a 的 CPU Python 双指针 → C 扩展 / NPU kernel）。
4. **序列并行 / 张量并行**（参考上游 `dist_ops/` + vLLM TP），把多卡显存真正用起来。

### 9.3 备选高风险高回报

5. **Triton-Ascend 主路径**（路径 B 重启）。前提是 CANN 8.2+ + torch_npu 2.6 + triton-ascend 三件套对齐。一旦可用，可直接复用上游 Triton kernel 大部分语义，并彻底解决 §8 / §3 / §9 三个根因。

### 9.4 长尾

6. 补 32K~64K 单卡 dense 实测，把"理论可跑"区间转为"实测可跑"。
7. 把 `torch_dtype=` 替换为 `dtype=`，清理 transformers 4.57 deprecation warning。
8. 与运维沟通 CANN 目录 owner 问题，消除 `torch_npu.utils.collect_env` 的 UserWarning。

---

## 10. 交付清单

- 代码主线：`MInference-NPU/`（独立 git repo，独立 `setup.py`，可 `pip install -e .`）
- 测试：`MInference-NPU/tests/`（5 个测试文件 + 2 个 M5 实测原始日志 md）
- HF 端到端示例：`MInference-NPU/examples/run_hf_minimal.py`
- 文档：
  - `MInference-NPU/docs/{SETUP,M1_dense_pipeline,M2_streaming,M3_block_sparse,M4_vertical_slash}.md`（实施细节）
  - `docs/context_checkpoint.md`（实时进度 + 关键决策点 + 踩坑链接）
  - `docs/migration_plan_v1.md`（立项方案 + 实施变更说明）
  - `docs/migration_v1_notes.md`（踩坑清单）
  - 本报告

---

## 附录 A：M5 实测原始日志索引

| 测试轮 | 文件 | 关键 case |
|---|---|---|
| 第一轮 | `MInference-NPU/tests/M5_test_results.md` | 8K minference / 32K 三方对比 / 128K 4卡 |
| 第二轮 | `MInference-NPU/tests/M5_test_results_round2.md` | 8K minference 复跑（排除 JIT） / 16K minference / 16K dense |
