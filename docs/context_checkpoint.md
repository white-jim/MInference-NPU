# 上下文检查点 — MInference 1.0 → 昇腾 NPU 算法迁移

> 用途：在新会话中快速恢复工作上下文。读这一份就能接着干，不必重新探索代码库。
> 创建日期：2026-05-23（最近更新：2026-05-23，已完成 M0 + M1 + **M2 步骤 1**（streaming kernel 文件落地），下一步：M2 步骤 2 —— 接入 `__init__.py` / `minference_forward.py` + 写单测 + 写文档）
> 工作目录：`D:\works\算法迁移\`

---

## 1. 任务背景

把 Microsoft 的 [MInference 1.0](https://github.com/microsoft/MInference) 长上下文 LLM 推理加速算法迁移到 **华为昇腾 NPU 平台**（v1 用 **Triton-Ascend** 路径）。

- 源码本体：`D:\works\算法迁移\MInference\`（git 仓库的本地拷贝，**不动**）
- v1 产出：`D:\works\算法迁移\MInference-NPU\`（独立目录，完整代码树）
- 文档：`D:\works\算法迁移\docs\`
  - `MInference_1.0_implementation.md` — MInference 1.0 实现详解
  - `ascend_migration_survey.md` — 昇腾平台 + MInference 可复用成果调研
  - `migration_plan_v1.md` — v1 迁移实施方案（M0–M5 五阶段）
  - `context_checkpoint.md` — 本文件

---

## 2. 已完成（按时序）

### 2.1 调研与方案阶段
- [x] 通读 MInference 仓库整体结构（多个子项目：MInference 1.0 / SCBench / LeanK / TriangleMix / MMInference / mtraining）
- [x] 输出 MInference 1.0 实现详解 `docs/MInference_1.0_implementation.md`（约 500 行）
- [x] 输出昇腾平台调研 `docs/ascend_migration_survey.md`（约 280 行，56 条引用）
- [x] 输出 v1 迁移实施方案 `docs/migration_plan_v1.md`（M0–M5 五阶段，4 项决策拍板）

### 2.2 M0 — 环境与骨架（2026-05-23，按假定 910B 配置完成代码与文档侧）
- `MInference-NPU/` 骨架建立：`minference/{ops,modules,configs,backend_npu}` + `tests/` + `examples/` + `docs/` + `setup.py`（无 CUDAExtension）
- 上游 v1 范围内 py 源已拷贝（patch / minference_forward / ops / configs / utils 等），剔除 dist_ops / v3 / dilated / static / tri_shape / kvcompress 等排除项
- `tests/test_env.py`：vector add Triton-Ascend kernel + `npu_fusion_attention` dense causal smoke，非 NPU 环境优雅 FAIL
- `docs/SETUP.md`：假定 910B 配置矩阵（Atlas 800T A2 / 8×910B3 64GB / Ubuntu 22.04 aarch64 / CANN 8.3.RC1 / torch 2.6.0 + torch_npu 2.6.0.RC1 / triton-ascend / Python 3.10 / transformers ≥ 4.45 / accelerate ≥ 0.28），含装机步骤、test_env.py 跑法、5 类常见踩坑、§6 实机回填 TODO

### 2.3 M1 — 上层 Python 链路打通（2026-05-23，dense fallback）
关键策略：**三种稀疏分支全部退化为 `backend_npu.dense_attention`**，per-head 循环骨架保留，M2/M3/M4 按分支逐个替换回真 NPU 稀疏 kernel。

- `minference/backend_npu/`（**新增**）
  - `attention.py`：`dense_attention` / `prefill_dense` / `decode_dense` 三个 `torch_npu.npu_fusion_attention` / `npu_incre_flash_attention` 薄封装；CPU/CUDA 环境下走纯 PyTorch eager 兜底；`is_npu_available()` 检测
  - `cuda_shim.py`：`convert_vertical_slash_indexes` 占位（M4-a 实现）
  - `__init__.py`：re-export 上述符号
- `minference/patch.py`（**重写**，~140 行）：monkey-patch `LlamaAttention.forward`；移除 vLLM / KV-CPU offload / KV 压缩 / inf_llm / chatglm 等 v1 排除项；`patch_hf` 兼容 API（仅接 `attn_type ∈ {minference, dense}`）
- `minference/models_patch.py`（**重写**，~80 行）：`MInference` 类仅接受 `attn_type ∈ {minference, dense, hf}` / `kv_type = dense`，其他全部 raise
- `minference/modules/minference_forward.py`（**重写**，~280 行）：
  - 三种稀疏分支（vertical_and_slash / stream_llm / block_sparse）全部退化为 `dense_attention(q, k, v, causal=True)`
  - 保留：per-head 循环骨架 + best_pattern 加载 + `sum_all_diagonal_matrix`（M4 复用）+ RoPE 类型探测
  - `device="cuda"` 全部改为 device-agnostic（跟随输入 tensor.device）
- `minference/__init__.py`（**重写**）：导出 `MInference / patch_hf / minference_patch` + 三个 sparse op 门面（`vertical_slash_sparse_attention` / `block_sparse_attention` / `streaming_forward`，签名与上游对齐，多余参数忽略，M2-M4 时按分支替换）
- 上游 `patch.py` / `models_patch.py` / `minference_forward.py` 保留为 `*_upstream.py`，**不被任何代码 import**，仅供 M2-M4 对照
- `tests/test_dense_forward.py`：3 个子测试（dense vs HF eager 数值一致 / per-head dense fallback / accelerate ≥ 2 卡 device_map="auto"，单卡环境跳过最后一个）
- `examples/run_hf_minimal.py`：最小 HF 调用脚本（Qwen2.5 / Llama-3.1 / 单卡或多卡 / 8k–32k+ ctx_len / 三种 attn_type 切换）
- `docs/M1_dense_pipeline.md`：M1 用法 / 验收标准 / 排查 / 与上游对照表 / M2 切换指引

**Windows 上 syntax 全通过**，实机测试待 NPU 服务器到位后跑。

### 2.4 M2 — Streaming kernel（**进行中**，2026-05-23 完成步骤 1）

**已完成（步骤 1：kernel 文件落地）**
- **新增** `minference/ops/streaming_kernel_npu.py`：
  - `streaming_forward(q, k, v, n_init, n_local)`：顶层入口，签名严格对齐上游 `streaming_kernel.py:streaming_forward`
  - head_dim 不在 `{16,32,64,128,256,512}` 时 pad 到 2 的幂，输出截回原始 head_dim（与上游一致）
  - 短路 `k_len <= n_local`：退化为 `backend_npu.dense_attention`（causal dense）
  - **NPU 路径** `_streaming_npu`：两段 `npu_fusion_attention` + log-sum-exp 合并
    - 段 1：sliding-window-only，`sparse_mode=4`（band），`pre_tockens=n_local-1`，`next_tockens=0`
    - 段 2：仅前 `n_init` 个 sink key，自定义 `atten_mask` 排除 sliding 重叠（complement-sliding-window 思路；mask shape `[1,1,S_q,n_init]`，长上下文下也很小）
    - 合并：`m_new = max(m1,m2)`；`o = (o1*l1*exp(m1-m)+o2*l2*exp(m2-m))/(l1*exp(m1-m)+l2*exp(m2-m))`；含 `l==0` 防御
    - 取 `softmax_max[..., :1]` / `softmax_sum[..., :1]` 处理 NPU FA 的 8-lane tile padding；兼容 3D 返回
    - `n_init <= 0` 或"段 2 全 mask"时单段返回 o1，避免 -inf 触发 NaN
  - **非 NPU 路径** `_streaming_pytorch_ref`：纯 PyTorch sink+sliding mask，按 q 分块控制 mask 在 ~4M 元素；fp32 softmax；同时作为单测黄金参考
- Windows `py_compile` 通过；本机无 torch，数值校验留到 M2 步骤 2 的单测里跑

**待办（步骤 2-4：接入、单测、文档）**
- [ ] `minference/__init__.py`：`streaming_forward = _dense` → `from .ops.streaming_kernel_npu import streaming_forward`，删掉 wrapper
- [ ] `minference/modules/minference_forward.py:gather_last_q_vertical_slash_topk_v4`：`stream_llm` 分支由 `dense_attention(...)` 改为 `streaming_forward(q, k, v, vertical_size, slash_size)`（上游用 `(vertical_size, slash_size)` 复用为 `(n_init, n_local)`，见上游 `minference_forward.py:481-486`）
- [ ] **新增** `tests/test_streaming_kernel.py`：多组 `(n_init, n_local)` + 三种 `k_len` 区段（短路 / 边界 / 长上下文），与 PyTorch ref 内嵌的"逐 row naive 实现"做 max_abs_diff < 1e-4 校验；NPU 上额外跑 `_streaming_npu` vs `_streaming_pytorch_ref` 对比 < 1e-2
- [ ] **新增** `docs/M2_streaming.md`：算法、接口、两段合并公式、测试跑法、与 M3 衔接

---

## 3. v1 决策速记（已拍板，写入 `docs/migration_plan_v1.md`）

| 维度 | 选择 |
|---|---|
| 迁移范围 | vertical_and_slash + block_sparse + stream_llm + dense fallback（**不含** dilated/static/tri_shape/kvcompress/dist_ops/vLLM 集成/FA3） |
| 算子主路径 | **Triton-Ascend** |
| `convert_vertical_slash_indexes` | **Triton-Ascend 重写**（实现遇阻可临时 host CPU fallback） |
| 框架宿主 | **HF transformers + torch_npu**（accelerate 多卡 `device_map="auto"` 在 v1 覆盖范围内） |
| 产出形态 | 独立 `MInference-NPU/` 目录（完整 py 源 + tests + examples + docs + 独立 setup.py） |

---

## 4. 进度与下一步

```
[x] 调研 + 方案拍板
[x] M0 环境与骨架（按假定 910B 配置；实机回填待跑）
[x] M1 上层 Python 链路打通（dense fallback）
[~] M2 Streaming kernel              ← 进行中（步骤 1/4 完成：kernel 文件落地）
[ ] M3 Block-Sparse kernel
[ ] M4-a convert_vertical_slash_indexes
[ ] M4-b Vertical-Slash 主 kernel（核心难点）
[ ] M5 端到端联调 + 精度/性能/效果
[ ] migration_v1_notes.md + migration_v1_report.md
```

**与实机校验解耦**：M2/M3/M4 的代码改造在 Windows 工作机上写、syntax 检查通过即可推进；实机跑 `test_env.py` / `test_dense_forward.py` / `test_*_kernel.py` 是用户拿到 NPU 机器后并行做的事，不卡迁移工作主线。

---

## 5. M2 入手要点（下一步直接接的事）

**目标**：把 `streaming_forward`（A-shape）从 dense fallback 替换为真稀疏调用，验证 `n_init=128 / n_local=3968` 默认配置语义等价。

### 5.0 当前进度（2026-05-23）
**步骤 1 已完成**：`minference/ops/streaming_kernel_npu.py` 落地（详见 §2.4）。
下一会话直接从下面"步骤 2/3/4" 接着写即可，不必重读上游 streaming_kernel.py。

### 5.1 实现路径（已采用 "两段 + 合并"）
NPU 版 `streaming_forward` 内部走两段 `npu_fusion_attention` + log-sum-exp 合并（plan §4 备选路径）：
- 段 1：`sparse_mode=4`（band），`pre_tockens=n_local-1`，`next_tockens=0` —— sliding window only
- 段 2：仅前 n_init 个 sink key + 自定义 `atten_mask` 排除 sliding 重叠（complement-sliding-window 思路）
- 跨段用 NPU FA 返回的 `softmax_max` / `softmax_sum` 做 LSE 合并

之所以不走"sparse_mode=4 一次调用"：`pre_tockens`/`next_tockens` 表达的是 band，不能同时表达"前 n_init 个 token 永远可见"的 sink 段。`prefix` 在不同 torch_npu 版本上语义不稳定，暂不依赖。

### 5.2 落点（具体改哪几个文件） — 步骤 2/3/4 待办
1. **改** `minference/__init__.py`：把 `streaming_forward = _dense` 改为 `from .ops.streaming_kernel_npu import streaming_forward`（去掉 wrapper）
2. **改** `minference/modules/minference_forward.py:gather_last_q_vertical_slash_topk_v4`：把 `if ty == "stream_llm": return dense_attention(...)` 分支改为 `return streaming_forward(q, k, v, vertical_size, slash_size)`（上游约定：`stream_llm` 分支 best_pattern 的 (v_size, s_size) 复用为 (n_init, n_local)，见 `MInference/minference/modules/minference_forward.py:481-486`）
3. **新增** `tests/test_streaming_kernel.py`：随机 q/k/v + 多组 `(n_init, n_local)`，与"PyTorch 黄金参考（手写 sink + sliding window mask + dense softmax）"差异 < 1e-2；NPU 上额外做 `_streaming_npu` vs `_streaming_pytorch_ref` 对照
4. **新增** `docs/M2_streaming.md`：算法说明、两段 LSE 合并公式、接口约定、测试跑法

### 5.3 验收（plan §4）
- `test_streaming_kernel.py` 全 PASS
- 固定 best_pattern 把所有 head 强制设为 `stream_llm` 的 case，输出与"该 pattern 在 CUDA 上的输出"差异 < 1e-2

### 5.4 关键参考
- 上游接口：`MInference/minference/ops/streaming_kernel.py:611` (`streaming_forward`) + `:680` (`stream_llm_forward`)
- 上游 stream_llm 分支调度：`MInference/minference/modules/minference_forward.py:474-486`
- v1 NPU kernel 实现：`MInference-NPU/minference/ops/streaming_kernel_npu.py`
- 调研对照表：`docs/ascend_migration_survey.md` Streaming 行 + plan §4 "关键点"段

---

## 6. 关键代码地图

### 6.1 上游 MInference（参考蓝本，**不动**）

| 角色 | 路径 |
|---|---|
| 顶层用户入口 | `MInference/minference/models_patch.py:19` (`MInference` 类) |
| HF monkey-patch | `MInference/minference/patch.py:1313` (`patch_hf`) |
| vLLM monkey-patch | `MInference/minference/patch.py` (`minference_patch_vllm*`，文件尾部) |
| **per-head 调度** | `MInference/minference/modules/minference_forward.py:296` (`gather_last_q_vertical_slash_topk_v4`) |
| **在线 vertical/slash 估计** | `MInference/minference/modules/minference_forward.py:381` (`vertical_and_slash_kernel`) |
| 反对角线求和（精妙处） | `MInference/minference/modules/minference_forward.py:110` (`sum_all_diagonal_matrix`) |
| **Vertical-Slash op wrapper** | `MInference/minference/ops/pit_sparse_flash_attention_v2.py:195` |
| **Vertical-Slash Triton kernel** | `MInference/minference/ops/pit_sparse_flash_attention_v2.py:48` |
| Block-Sparse op + kernel | `MInference/minference/ops/block_sparse_flash_attention.py:29` / `:169` |
| Streaming (A-shape) kernel | `MInference/minference/ops/streaming_kernel.py:26` |
| **CUDA 索引展开 kernel** | `MInference/csrc/vertical_slash_index.cu:27` |
| pybind 导出 | `MInference/csrc/kernels.cpp:7` |
| Best-pattern JSON 示例 | `MInference/minference/configs/Llama_3.1_8B_Instruct_128k_kv_out_v32_fit_o_best_pattern.json` |

### 6.2 NPU v1 产出（开发主线）

| 角色 | 路径 | 状态 |
|---|---|---|
| 顶层入口 | `MInference-NPU/minference/models_patch.py` (`MInference` 类) | M1 ✓ |
| HF monkey-patch | `MInference-NPU/minference/patch.py` (`patch_hf` / `minference_patch`) | M1 ✓ |
| Per-head 调度 + 顶层 forward | `MInference-NPU/minference/modules/minference_forward.py` | M1 ✓（三 pattern dense） |
| NPU dense 薄封装 | `MInference-NPU/minference/backend_npu/attention.py` | M1 ✓ |
| `convert_vertical_slash_indexes` 占位 | `MInference-NPU/minference/backend_npu/cuda_shim.py` | M4-a 待实现 |
| **Streaming kernel** | `MInference-NPU/minference/ops/streaming_kernel.py`（上游拷贝） | **M2 待重写** → `streaming_kernel_npu.py` |
| Block-Sparse kernel | `MInference-NPU/minference/ops/block_sparse_flash_attention.py`（上游拷贝） | M3 待重写 |
| Vertical-Slash kernel | `MInference-NPU/minference/ops/pit_sparse_flash_attention_v2.py`（上游拷贝） | M4-b 待重写 |
| 上游参考（不 import） | `MInference-NPU/minference/{patch_upstream,models_patch_upstream}.py` + `modules/minference_forward_upstream.py` | M2-M4 时对照 |
| Best-pattern JSON | `MInference-NPU/minference/configs/*.json` | 直接复用上游 |
| 环境烟测 | `MInference-NPU/tests/test_env.py` | M0 ✓ |
| Dense forward 测试 | `MInference-NPU/tests/test_dense_forward.py` | M1 ✓ |
| HF 调用示例 | `MInference-NPU/examples/run_hf_minimal.py` | M1 ✓ |
| 模块文档 | `MInference-NPU/docs/{SETUP,M1_dense_pipeline}.md` | M0/M1 ✓ |

---

## 7. MInference 1.0 速记（关键事实，迁移时反复用到）

- **加速对象**：仅 prefill 阶段；decoding (`q_len == 1`) 短路成 dense flash_attn。
- **三种稀疏模式**：`vertical_and_slash`（主力，~95% head）、`block_sparse`、`stream_llm`（A-shape）。
- **离线**：每 (layer, head) 一组 `(pattern_type, v_size, s_size, score)`，写在 `configs/*.json`。
  - 常见默认：`vertical_and_slash, v=1000, s=6096`。
- **在线估计**：仅用 **最后 64 个 query** 与全 K 做一次 QK，softmax 后：
  - 列和 → vertical 列重要性 → topk → `v_idx`
  - 反对角线和 → slash 行重要性 → topk → `s_idx`
  - 强制保留前 30 列（sink）和最近 100 条 slash（local），避免退化。
- **CUDA 索引展开**：双指针扫 `v_idx` (升序) + `s_idx` (降序)，**合并相邻 slash** 为连续 range，**vertical 去重**（落在 range 内不重复），输出 `(block_count, block_offset, column_count, column_index)` 喂给 Triton kernel。
- **Triton kernel 两段循环**：
  1. 段 1：按 `block_offset` 扫 slash 拼出的连续 K 块（访存友好）。
  2. 段 2：按 `column_index` gather vertical 零散列。
  3. 两段共享 `m_i / l_i` 做 online softmax，等价完整 softmax。
- **per-head 循环**：不同 head 选不同 pattern，无法 batch；本身开销远小于稀疏 kernel 省下的时间。
- **starting_layer**：前若干层走 dense（pattern 还没收敛）。
- **构建产物**：上游 `minference.cuda` 子模块（`setup.py` 的 `CUDAExtension` 编译 `csrc/{kernels.cpp, vertical_slash_index.cu}`）；NPU v1 **不构建**此扩展，由 `backend_npu/cuda_shim.py` 占位。

---

## 8. 调研结论速记（来自 `ascend_migration_survey.md`）

**没有现成的 MInference 1.0 昇腾整套移植**。4 个核心组件每一个都有可复用资源：

| MInference 1.0 组件 | 昇腾可复用 | v1 工作量 | 落到哪个 milestone |
|---|---|---|---|
| Dense FA (fallback / decode) | `torch_npu.npu_fusion_attention` / `npu_prompt_flash_attention` / `npu_incre_flash_attention` / `npu_fused_infer_attention_score` | **零** | M1 ✓ |
| Streaming (A-shape) | `npu_fusion_attention` 的 `sparse_mode=4` band-causal，一行替换 | 极低 | **M2** |
| Block-Sparse FA | `npu_fusion_attention` 的 sparse_mode + atten_mask，或 TileLang `examples/blocksparse_attention/` | 低 | M3 |
| **Vertical-Slash FA**（主路径） | **TileLang-Ascend `examples/sparse_flash_attention/`** + **cann-recipes-infer `npu_sparse_flash_attention.py`**（DeepSeek-V3.2 DSA 参考） | **中-高** | M4-b |
| `convert_vertical_slash_indexes` | 无 1-to-1；参考 vllm-ascend RFC #5507 `hamming_dist_top_k` / RFC #807 `advance_step` | 中 | M4-a |
| per-head 调度 + best_pattern JSON | 纯 Python，**直接复用** | 零 | M1 ✓ |

**学术蓝本**：
- **FastAttention** (arXiv:2410.16663) — FA2 → Ascend 完整迁移，two-level tiling / tiling-mask / tiling-AllReduce
- **AMLA** (arXiv:2509.25224) — 已并入 CANN，Ascend 910 上 86.8% 峰值

**框架宿主推荐路线**：HF transformers + torch_npu（v1） → vLLM-Ascend（生产化，v2）。

---

## 9. 环境信息

- **工作目录**：`D:\works\算法迁移\`（**不是** git repo）
- **子目录**：
  - `MInference/` — 上游源码本体（参考蓝本，v1 不动）
  - `MInference-NPU/` — v1 产出（M0 + M1 完成；M2 起继续扩展 `ops/streaming_kernel_npu.py` 等）
  - `docs/` — 本系列方案 / 调研 / 实现详解 / 检查点文档
- **平台**：Windows 10 Pro / bash shell / Unix 风格路径
- **注意**：源码目录路径含中文"算法迁移"，shell 操作需引号包裹
- **目标平台关键版本约束**：CANN ≥ 8.3.RC1、torch-npu ≥ 2.6.0.RC1、accelerate ≥ 0.28（多卡 device_map 识别 NPU 需要此版本起）
- **假定的目标服务器配置**（待实机确认）：Atlas 800T A2 / 8×910B3 64GB HBM / Ubuntu 22.04 aarch64 / CANN 8.3.RC1 / torch 2.6.0 + torch_npu 2.6.0.RC1 / triton-ascend / Python 3.10 / transformers ≥ 4.45 / accelerate ≥ 0.28。完整矩阵见 `MInference-NPU/docs/SETUP.md`
- **实机校验脚本**（用户拿到 NPU 机器后跑）：
  - `python MInference-NPU/tests/test_env.py` — M0 烟测（vector add + npu_fusion_attention）
  - `python MInference-NPU/tests/test_dense_forward.py` — M1 dense fallback 数值验证 + 多卡
  - `python MInference-NPU/examples/run_hf_minimal.py` — 真模型 8k–32k+ smoke

---

## 10. v1 多卡支持（accelerate）速记

- ✅ **HF accelerate 自动多卡部署**（`device_map="auto"` 按 layer 切 NPU）：**v1 直接覆盖**。原理：accelerate 在 layer 边界切，attention 在层内闭环跑，对算子透明
- ❌ **vLLM TP / 序列并行 / 显式 DP / EP**：v1 不做，留 v2
- **实现约束**（M1 已落地，M2-M4 继续守住）：
  1. device-agnostic（禁止写死 `npu:0`、所有 tensor 跟随输入 device、Triton-Ascend kernel launch 跟随输入）
  2. accelerate ≥ 0.28
  3. 长上下文场景手动给 `max_memory` 留 KV 余量（配置层面解决，非代码改动）
