# 上下文检查点 — MInference 1.0 → 昇腾 NPU 算法迁移

> 用途：在新会话中快速恢复工作上下文。读这一份就能接着干，不必重新探索代码库。
> 创建日期：2026-05-23（最近更新：2026-05-23，已完成 M0 + M1 + M2 + M3 + **M4 全部步骤**（convert_vertical_slash_indexes CPU Python + vertical-slash NPU kernel + 接入 + 单测 + 文档），下一步：M5 端到端联调 + 精度/性能/效果）
> 工作目录：`D:\tempt\算法迁移\`（本次会话）

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

### 2.4 M2 — Streaming kernel（**已完成**，2026-05-23）



**步骤 1（kernel 文件落地）**
- **新增** `minference/ops/streaming_kernel_npu.py`：
  - `streaming_forward(q, k, v, n_init, n_local)`：顶层入口，签名严格对齐上游 `streaming_kernel.py:streaming_forward`
  - head_dim 不在 `{16,32,64,128,256,512}` 时 pad 到 2 的幂，输出截回原始 head_dim（与上游一致）
  - 短路 `k_len <= n_local`：退化为 `backend_npu.dense_attention`（causal dense）
  - **NPU 路径** `_streaming_npu`：两段 `npu_fusion_attention` + log-sum-exp 合并
    - 段 1：sliding-window-only，`sparse_mode=4`（band），`pre_tockens=n_local-1`，`next_tockens=0`
    - 段 2：仅前 `n_init` 个 sink key，自定义 `atten_mask` 排除 sliding 重叠；mask shape `[1,1,S_q,n_init]`
    - 合并：`m_new = max(m1,m2)`；`o = (o1*l1*exp(m1-m)+o2*l2*exp(m2-m))/(l1*exp(m1-m)+l2*exp(m2-m))`；含 `l==0` 防御
    - `n_init <= 0` 或"段 2 全 mask"时单段返回 o1，避免 -inf 触发 NaN
  - **非 NPU 路径** `_streaming_pytorch_ref`：纯 PyTorch sink+sliding mask，按 q 分块控制内存；fp32 softmax；同时作为单测黄金参考

**步骤 2（接入）**
- `minference/__init__.py`：删掉 dummy wrapper，改为 `from .ops.streaming_kernel_npu import streaming_forward`
- `minference/modules/minference_forward.py`：`stream_llm` 分支改为 `_streaming_forward(q, k, v, n_init=vertical_size, n_local=slash_size)`（上游约定 vertical_size→n_init, slash_size→n_local）；block_sparse/vertical_and_slash 仍保留 dense（M3/M4 替换）

**步骤 3（单测）**
- **新增** `tests/test_streaming_kernel.py`：5 组测试（ref vs naive / 顶层 vs naive 含短路 / head_dim pad / NPU vs ref / 参数扫描），CPU 容差 1e-3，fp16/NPU 容差 1e-2

**步骤 4（文档）**
- **新增** `docs/M2_streaming.md`：算法背景、两段路径说明、LSE 合并公式、接口约定、调用链、测试跑法、与 M3 衔接、与上游差异对照表

### 2.5 M3 — Block-Sparse kernel（**已完成**，2026-05-23）

**步骤 1（kernel 文件落地）**
- **新增** `minference/ops/block_sparse_kernel_npu.py`：
  - `block_sparse_attention(q, k, v, topk_blocks, block_size=64)`：顶层入口，签名对应上游 `top_k` 参数
  - head_dim 不在 `{16,32,64,128,256,512}` 时 pad 到 2 的幂，输出截回原始 head_dim
  - 序列长度 > `_MAX_SEQ_FOR_MASK=16384` 时退化为 `dense_attention` + WARNING（路径 B 超长序列）
  - **NPU 路径** `_block_sparse_npu`：`_build_block_sparse_mask` + `npu_fusion_attention(sparse_mode=1, atten_mask=mask)`
    - `_build_block_sparse_mask`：mean-pool Q/K → block 级 QK top-k → `block_attend [B,H,n_bq,n_bk]`
      → expand+reshape → `token_attend [B,H,S_q,S_k]` → 追加 per-token 因果约束 → True=masked
    - 关键正确性保证：per-token 因果 mask 消除 block top-k 中 -inf tie-breaking 引入的误差
  - **非 NPU 路径** `_block_sparse_pytorch_ref`：共用 `_build_block_sparse_mask`，PyTorch masked softmax；同时作为单测黄金参考

**步骤 2（接入）**
- `minference/modules/minference_forward.py`：新增 `block_sparse` 分支 `return _block_sparse_attention(q, k, v, topk_blocks=int(vertical_size))`（`vertical_size` 对应上游 `top_k`）
- `minference/__init__.py`：删掉 dummy wrapper，改为 `from .ops.block_sparse_kernel_npu import block_sparse_attention`

**步骤 3（单测）**
- **新增** `tests/test_block_sparse_kernel.py`：6 组测试（mask 因果等价 / mask 因果不变性 / PyTorch ref vs dense / shape/dtype / head_dim pad / NPU vs ref / 参数扫描），CPU 容差 1e-4，NPU 容差 1e-2

**步骤 4（文档）**
- **新增** `docs/M3_block_sparse.md`：算法背景、mask 构建 6 步、reshape 合法性验证、NPU 路径代码、接口约定、调用链、序列长度限制与路径 B、测试跑法、与上游差异对照表、与 M4 衔接

### 2.6 M4 — Vertical-Slash kernel（**已完成**，2026-05-23）

**步骤 1 M4-a（`convert_vertical_slash_indexes` CPU Python）**
- **改** `minference/backend_npu/cuda_shim.py`：实现 `convert_vertical_slash_indexes`
  - 纯 Python + PyTorch CPU，签名与上游 pybind export 对齐（`causal` 参数额外保留）
  - 双指针算法：升序 `v_idx` + 降序 `s_idx`，合并相邻 slash 为连续 range，vertical 去重
  - `_process_block`：单 (batch, head, query_block) 的核心逻辑，对应 CUDA `convert_vertical_slash_indexes_kernel`
  - `_save_blocks`：对应 CUDA `save_blocks` device 函数

**步骤 1 M4-b（Vertical-Slash NPU kernel）**
- **新增** `minference/ops/vertical_slash_kernel_npu.py`：
  - `vertical_slash_sparse_attention(q, k, v, v_idx, s_idx, ...)` 顶层入口，签名与上游 `pit_sparse_flash_attention_v2.py` 对齐
  - head_dim pad（与 M2/M3 一致）；序列长度 > `_MAX_SEQ_FOR_MASK=16384` 退化 dense + WARNING
  - `_build_vs_mask_from_indexes`：从 M4-a 输出构建 token 级 bool mask
    - cumsum 区间标记法构建 slash 覆盖：O(NNZ_S + S_k) 而非 O(NNZ_S × S_k)
    - scatter set 构建 vertical 覆盖；广播因果约束 j≤i；mask AND 写入
  - **NPU 路径** `_vertical_slash_npu`：CPU 侧 M4-a + 构建 mask → `npu_fusion_attention(sparse_mode=1)`
  - **非 NPU 路径** `_vertical_slash_pytorch_ref`：同 CPU 侧 M4-a + mask → PyTorch masked softmax

**步骤 2（接入）**
- `minference/modules/minference_forward.py`：
  - 新增 `_vertical_and_slash_kernel(self, q, k, v, vertical_size, slash_size)`：在线估计 v_idx/s_idx（列和 topk + 反对角线和 topk），调用 `_vs_sparse_attention`
  - `gather_last_q_vertical_slash_topk_v4` 的 `vertical_and_slash` 分支替换为 `_vertical_and_slash_kernel`（原 dense fallback 注释移除）
  - 新增 `from ..ops.vertical_slash_kernel_npu import vertical_slash_sparse_attention as _vs_sparse_attention`
- `minference/__init__.py`：删掉 dummy `vertical_slash_sparse_attention` wrapper，改为 `from .ops.vertical_slash_kernel_npu import vertical_slash_sparse_attention`；移除已无用的 `_dense` 别名

**步骤 3（单测）**
- **新增** `tests/test_vertical_slash_kernel.py`：8 组测试（convert 格式/因果/计数 / mask 形状/边界/因果 / ref vs dense 满覆盖 / sparse shape / head_dim pad / NPU vs ref / 参数扫描 / dtype），CPU 容差 1e-3 / 5e-2（fp16），NPU 容差 1e-2

**步骤 4（文档）**
- **新增** `docs/M4_vertical_slash.md`：算法背景、双指针 M4-a 算法、cumsum mask 构建 M4-b、接口约定、调用链、序列长度限制、测试跑法、与上游差异对照表、与 M5 衔接

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
[x] M2 Streaming kernel              ← 已完成（步骤 1-4 全部完成）
[x] M3 Block-Sparse kernel           ← 已完成（步骤 1-4 全部完成）
[x] M4-a convert_vertical_slash_indexes  ← 已完成（CPU Python 双指针，与 CUDA kernel 等价）
[x] M4-b Vertical-Slash 主 kernel    ← 已完成（cumsum mask + npu_fusion_attention + 在线估计接入）
[ ] M5 端到端联调 + 精度/性能/效果
[ ] migration_v1_notes.md + migration_v1_report.md
```

**与实机校验解耦**：M2/M3/M4 的代码改造在 Windows 工作机上写、syntax 检查通过即可推进；实机跑 `test_env.py` / `test_dense_forward.py` / `test_*_kernel.py` 是用户拿到 NPU 机器后并行做的事，不卡迁移工作主线。

---

## 5. M5 入手要点（下一步直接接的事）

**目标**：端到端联调，在 NPU 机器上验证 M2–M4 三种稀疏 kernel 的精度与性能。

### 5.1 M5 概述

M5 不引入新代码，主要是实机验证：

| 验证项 | 命令 |
|---|---|
| 环境烟测 | `python tests/test_env.py` |
| Dense pipeline | `python tests/test_dense_forward.py` |
| Streaming kernel | `python tests/test_streaming_kernel.py` |
| Block-sparse kernel | `python tests/test_block_sparse_kernel.py` |
| Vertical-slash kernel | `python tests/test_vertical_slash_kernel.py` |
| HF 端到端 | `python examples/run_hf_minimal.py --attn_type minference ...` |

### 5.2 产出
- `docs/migration_v1_notes.md`：实测踩坑记录（NPU API 差异、精度偏差等）
- `docs/migration_v1_report.md`：v1 迁移报告（精度 / 性能 / 效果三维对比）

### 5.3 关键参考
- 上游 CUDA kernel：`MInference-NPU/MInference/minference/ops/op_utils/vertical_slash_utils.py`
- 上游 vertical_and_slash 调用链：`MInference-NPU/MInference/minference/modules/minference_forward.py:381`

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
| Per-head 调度 + 顶层 forward | `MInference-NPU/minference/modules/minference_forward.py` | M1 ✓，M2/M3 已接入 |
| NPU dense 薄封装 | `MInference-NPU/minference/backend_npu/attention.py` | M1 ✓ |
| `convert_vertical_slash_indexes` | `MInference-NPU/minference/backend_npu/cuda_shim.py` | **M4-a ✓**（CPU Python 双指针） |
| **Streaming kernel** | `MInference-NPU/minference/ops/streaming_kernel_npu.py`（NPU 实现） | **M2 ✓** |
| **Block-Sparse kernel** | `MInference-NPU/minference/ops/block_sparse_kernel_npu.py`（NPU 实现） | **M3 ✓** |
| **Vertical-Slash kernel** | `MInference-NPU/minference/ops/vertical_slash_kernel_npu.py`（NPU 实现） | **M4-b ✓** |
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
