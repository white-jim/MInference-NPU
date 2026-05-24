# 上下文检查点 — MInference 1.0 → 昇腾 NPU 算法迁移

> 用途：在新会话中快速恢复工作上下文。读这一份就能接着干，不必重新探索代码库。
> 创建日期：2026-05-23（最近更新：2026-05-24，已完成 M0 + M1 + M2 + M3 + **M4 全部步骤**（convert_vertical_slash_indexes CPU Python + vertical-slash NPU kernel + 接入 + 单测 + 文档）；**Bug 修复已完成**：B1/B2/B3/B4/B5/P3/P4 全部修正（见 §11）；**新增静态审查问题**：B6/B7/B8/B9/P5/P6/P7/P8/P9 待处理（见 §12），下一步：先修 B6/B7/B8，再进入 M5 端到端联调）
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
[x] Bug 修复（B1/B2/B3/B4/B5/P3/P4 全部修正，2026-05-24）
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

---

## 11. 代码审查结果（2026-05-23）

> 对 M0–M4 全部产出文件的全量审查，重点：NPU 上必现崩溃、潜在崩溃、性能大坑。
> 修复状态初始均为 `[ ]`，修复后改为 `[x]`。

### 11.1 必现崩溃（NPU 上 100% 报错）

#### B1 — `_build_vs_mask_from_indexes`：CPU 张量做 NPU 张量的 index
- **文件**：`minference/ops/vertical_slash_kernel_npu.py:128-157`
- **根因**：`convert_vertical_slash_indexes`（`cuda_shim.py`）输出的 `block_offset`、`column_index` 全是 **CPU tensor**（创建时没有指定 device）。但 `_build_vs_mask_from_indexes` 接收 `device=q.device`（NPU），在内部创建 NPU 张量后，用 CPU 张量做 index：
  ```python
  blk_starts = block_offset[b, h, bq, :nblk].long()      # CPU tensor
  cov_delta = torch.zeros(S_k + 1, dtype=torch.int32, device=device)  # NPU tensor
  cov_delta.scatter_add_(0, blk_starts.clamp(0, S_k), ...)  # CRASH: index on CPU, self on NPU

  cols = column_index[b, h, bq, :ncol].long().clamp(0, S_k - 1)  # CPU tensor
  vert_cov[cols] = True   # CRASH: index on CPU, vert_cov on NPU
  ```
- **修复**：使用前加 `.to(device)`：
  ```python
  blk_starts = block_offset[b, h, bq, :nblk].long().to(device)
  blk_ends   = (blk_starts + block_size).clamp(max=S_k)
  cols = column_index[b, h, bq, :ncol].long().to(device).clamp(0, S_k - 1)
  ```
- **状态**：`[x]` 已修复（2026-05-24）

#### B2 — `decode_dense`：`npu_incre_flash_attention` 返回值未解包
- **文件**：`minference/backend_npu/attention.py:167-174`
- **根因**：`npu_incre_flash_attention` 返回元组 `(output, ...)` 而非单个 tensor，但函数直接 return，没有像 `dense_attention` 一样做 `result[0] if isinstance(result, (tuple, list)) else result` 解包。上游拿到 tuple 后 shape 操作必然报错。
- **修复**：
  ```python
  result = torch_npu.npu_incre_flash_attention(q, k_cache, v_cache, ...)
  return result[0] if isinstance(result, (tuple, list)) else result
  ```
- **状态**：`[x]` 已修复（2026-05-24）

#### B3 — `seqlens` 硬编码 batch=1，B>1 时越界
- **文件**：`minference/ops/vertical_slash_kernel_npu.py:189`（`_vertical_slash_pytorch_ref`）和 `:234`（`_vertical_slash_npu`）
- **根因**：两处都写死 `seqlens = torch.tensor([S_k], dtype=torch.int32)`，shape 为 `[1]`。但 `convert_vertical_slash_indexes` 中 `seq_all = seqlens.cpu().tolist()` 然后 `for b in range(batch_size): seqlen = int(seq_all[b])`。B > 1 时 `seq_all[1]` 越界。
- **修复**：
  ```python
  seqlens = torch.tensor([S_k] * B, dtype=torch.int32)
  ```
- **状态**：`[x]` 已修复（2026-05-24）

---

### 11.2 潜在崩溃（特定条件下触发）

#### B4 — `streaming_forward` / `block_sparse_attention` 强断言 q/k/v shape 完全一致
- **文件**：`streaming_kernel_npu.py:313-315`，`block_sparse_kernel_npu.py:276-278`
- **根因**：`assert q.shape == k.shape == v.shape`。有 KV cache 时（k_len > q_len），k 和 v shape 与 q 不一致，AssertionError。`vertical_slash_kernel_npu.py` 中已正确分离 S_q / S_k，可参照。
- **修复**：把断言改为仅检查 dim=4 和 B/H/D 三维：
  ```python
  assert q.dim() == 4
  assert q.shape[0] == k.shape[0] and q.shape[1] == k.shape[1] and q.shape[-1] == k.shape[-1]
  ```
- **状态**：`[x]` 已修复（2026-05-24）

#### B5 — `atten_mask` 传 `bool` dtype，部分 torch_npu 版本要求 `uint8`
- **文件**：`block_sparse_kernel_npu.py:235`，`vertical_slash_kernel_npu.py:250`，`streaming_kernel_npu.py:255`
- **根因**：三个 kernel 均将 `torch.bool` mask 传给 `npu_fusion_attention(sparse_mode=1, atten_mask=mask)`。torch_npu 部分版本的 `npu_fusion_attention` 对 `atten_mask` 的 dtype 有严格要求（需 `uint8` 而非 `bool`），传 `bool` 会 TypeError。
- **修复**：三处 NPU 调用均先按 `torch.bool` mask 调用；若 torch_npu 小版本因 dtype 抛 `TypeError`，自动用 `mask.to(torch.uint8)` 重试。
- **状态**：`[x]` 已修复（2026-05-24）

---

### 11.3 重大性能问题

#### P1 — `_build_vs_mask_from_indexes`：三层 Python 循环内批量 NPU dispatch
- **文件**：`minference/ops/vertical_slash_kernel_npu.py:113-158`
- **根因**：`for b × for h × for bq` 共 B×H×num_rows 次 Python 迭代，每次内部执行多个 NPU 内核（`scatter_add_`、`cumsum`、bool 运算）。对典型 16k 序列（`_MAX_SEQ_FOR_MASK` 阈值）：1×32×256 = 8192 次迭代 × ~4 次 NPU dispatch = ~32k NPU 调度，每次有 Python→CANN 固定开销（10–100 μs），合计 **320ms–3.2s** 的纯 dispatch 损耗，远超实际计算。
- **修复方向**：将整个循环向量化——把 `block_offset`、`column_index` 全量搬到 NPU，用 batch 化的 scatter/cumsum 一次完成所有 block 的 coverage 计算，消除 Python 循环。
- **状态**：`[ ]`（需较大重构）

#### P2 — `convert_vertical_slash_indexes`：三层 Python 循环 × 纯 Python while 扫描
- **文件**：`minference/backend_npu/cuda_shim.py:205-232`
- **根因**：`for b × for h × for block_idx_m`，最内层 `_process_block` 是纯 Python while 循环，每次扫描全部 NNZ_V + NNZ_S（典型值 ~7000）。总操作量：32 × 2000 × 7000 ≈ **4.5 亿次 Python 操作**（128k 序列），估算耗时 **30–60 秒**。注释"计算量极小"是针对单 head 单次调用，忽略了三重循环规模。
- **修复方向**：用 NumPy/Cython 向量化替代纯 Python while 循环，或引入 Triton-Ascend kernel（方案 B）。短期可先用 NumPy 批量处理 v_list/s_list 的双指针逻辑，减少 Python 层级。
- **状态**：`[ ]`（需较大重构）

#### P3 — `init_minference_parameters` 每次 forward 重读 JSON + 句柄泄漏
- **文件**：`minference/modules/minference_forward.py:133-159`
- **根因**：`init_minference_parameters` 在每次 `forward` 开头无条件调用，内部用裸 `open(file)` 打开同一文件**两次**（一次 `len(json.load(open(...)))`，一次 `json.load(open(...))[...]`），文件句柄不关闭。32 层模型每次推理 = 64 次文件 I/O + 64 个泄漏句柄。
- **修复**（最容易，2 行）：在函数顶部加一次性初始化保护：
  ```python
  def init_minference_parameters(self) -> None:
      if getattr(self, '_minference_initialized', False):
          return
      ...  # 原有逻辑，open 改成 with open(...) as f: data = json.load(f)
      self._minference_initialized = True
  ```
- **状态**：`[x]` 已修复（2026-05-24）

#### P4 — `attn_mask.all().item()` 强制 NPU-CPU 同步
- **文件**：`minference/ops/streaming_kernel_npu.py:244`
- **根因**：`if bool(attn_mask.all().item()):` 中 `attn_mask` 在 NPU 上，`.item()` 强制将结果从 NPU 搬回 CPU，阻塞 NPU 异步执行流水线，每次推理必然产生一次 NPU-CPU 同步等待。
- **修复**：删除 `attn_mask.all().item()` 同步判断；在当前外层已保证 `s_k > n_local` 且 `n_init > 0` 的主路径下，该整张 mask 全 True 分支不可达。为防 torch_npu 对全 mask 行返回 NaN，保留 `l2 = torch.nan_to_num(l2, nan=0.0)`，LSE 合并会自然退化为 `o1`。
- **状态**：`[x]` 已修复（2026-05-24）

---

### 11.4 修复优先级

| 优先级 | 编号 | 估计改动量 | 说明 |
|---|---|---|---|
| P0（立刻改） | B1 | 3 行 | NPU 上每次都崩溃，M5 无法跑通 |
| P0（立刻改） | B2 | 2 行 | decode 路径必崩 |
| P0（立刻改） | B3 | 1 行 | B>1 必崩（推理通常 B=1，但测试可能触发） |
| P1（较早改） | P3 | ~10 行 | 最容易改的性能问题，每次推理必触发 |
| P1（较早改） | B4 | 4 行 | 有 KV cache 时触发，KV cache 场景常见 |
| P1（较早改） | P4 | 3 行 | 热路径同步，简单修复 |
| P1（已修复） | B5 | 3 处 | 已加 bool→uint8 TypeError retry，实机仍需确认最佳 dtype |
| P3（性能优化阶段） | P1 | 大重构 | 向量化 mask 构建，M5 性能测试后再做 |
| P3（性能优化阶段） | P2 | 大重构 | 向量化 convert_vertical_slash_indexes |

---

## 12. 代码审查结果（2026-05-24，第二轮静态审查）

> 审查范围：`MInference-NPU/MInference-NPU/minference/backend_npu/attention.py`、`backend_npu/cuda_shim.py`、`modules/minference_forward.py`、`ops/*_kernel_npu.py`、`patch.py`、`models_patch.py`、`tests/test_dense_forward.py` 与 M2/M3/M4 文档。
> 本轮仅做 Windows 开发机上的静态审查与 UTF-8 syntax check；由于本机无 torch/NPU 环境，未宣称实机跑通。
> Syntax check 通过：`attention.py / cuda_shim.py / minference_forward.py / streaming_kernel_npu.py / block_sparse_kernel_npu.py / vertical_slash_kernel_npu.py / patch.py / models_patch.py / __init__.py`。

### 12.1 高优先级崩溃 / 端到端不可用风险

#### B6 — 新版 Transformers attention forward / RoPE 接口不兼容
- **文件**：`minference/modules/minference_forward.py:81`、`:158`、`:284`、`:331`
- **根因**：当前 patched `forward()` 仍采用旧式 LlamaAttention 签名，并直接依赖 `self.rotary_emb`。较新版 transformers 的 Llama/Qwen attention 可能由外层传入 `position_embeddings=(cos, sin)`，且 `self_attn` 内不再持有 `self.rotary_emb`。
- **触发条件**：使用较新版 transformers（尤其 `position_embeddings` 路径）加载 Llama/Qwen/Mistral family 后调用 patched forward。
- **后果**：首次 forward 可能 `TypeError`（缺少旧式位置参数）或 `AttributeError: ... has no attribute rotary_emb`。
- **修复建议**：
  ```python
  def forward(..., position_embeddings=None, cache_position=None, **kwargs):
      if position_embeddings is not None:
          cos, sin = position_embeddings
      else:
          # 仅旧版 HF 走 self.rotary_emb 探测与 get_cos_sin
  ```
  同时无 `self.rotary_emb` 时不要在 `init_minference_parameters()` 中导入其模块的 `apply_rotary_pos_emb`。
- **状态**：`[ ]`

#### B7 — `attn_type="dense"` 未真正强制 dense，误入 vertical/slash 稀疏路径
- **文件**：`minference/models_patch.py:93-96`、`minference/patch.py:75-76`、`minference/modules/minference_forward.py:254-268`、`:374-380`
- **根因**：`models_patch.py` 注释称 dense 模式所有 head 走 dense，但 `patch.py` 只注入 `starting_layer/config_path`，未注入 `attn_type`。`dense` 默认 `starting_layer=-1`，因此 prefill 会进入 `gather_last_q_vertical_slash_topk_v4()`；无 best_pattern 时又默认 `("vertical_and_slash", 1000, 6096, 1)`。
- **触发条件**：`MInference(attn_type="dense")(model)`，包括 `tests/test_dense_forward.py:100` 和多卡 smoke。
- **后果**：dense smoke 不再是裸 dense；会构建 vertical/slash 索引与 token mask，可能引入 OOM、NPU sparse mask 报错或严重变慢；测试语义也不对。
- **修复建议**：在 `minference_patch()` 中注入 `model.config.attn_type = config.attn_type`；`init_minference_parameters()` 读取 `self.attn_type`；`forward()` 在 prefill 阶段优先判断 `attn_type == "dense"`，直接调用 `dense_attention()`，不要进入 per-head sparse 调度。
- **状态**：`[ ]`

#### B8 — Streaming NPU 路径假设 `npu_fusion_attention` 一定返回 softmax 统计量
- **文件**：`minference/ops/streaming_kernel_npu.py:220-222`、`:264-266`
- **根因**：`_streaming_npu()` 直接索引 `pass1[1]/pass1[2]` 与 `pass2[1]/pass2[2]`，默认返回 tuple/list 且包含 `softmax_max/softmax_sum`。不同 torch_npu 小版本可能只返回 output tensor，或 tuple 长度/语义不同。
- **触发条件**：目标 NPU 环境的 `torch_npu.npu_fusion_attention` 返回格式与当前假设不一致。
- **后果**：`IndexError`、把 tensor 第 0 维误当返回项导致 shape 错，或 LSE 合并崩溃。
- **修复建议**：显式校验 `isinstance(result, (tuple, list)) and len(result) >= 3`；若不可用，fallback 到一次性 user-mask attention、dense fallback 或直接抛带版本信息的清晰错误。
- **状态**：`[ ]`

#### B9 — `q_len < k_len` 的 chunked prefill / cache 场景 causal offset 不一致
- **文件**：`minference/modules/minference_forward.py:212-238`、`minference/ops/vertical_slash_kernel_npu.py:154`、`minference/ops/block_sparse_kernel_npu.py:139-162`
- **根因**：streaming 路径明确用 `abs_i = S_k - S_q + i`，但 vertical/slash mask 使用 `j <= q_rows`，block-sparse block 级 causal 使用 `bq >= bk`，均未统一加入 KV cache offset。`_vertical_and_slash_kernel()` 中 `s_idx = (q_len - 1) - s_raw_idx` 在 `S_k != q_len` 时语义也可疑。
- **触发条件**：full prefill 以外的 chunked prefill、prefill with past、或任何 `S_q != S_k` 且 `q_len != 1` 的路径。
- **后果**：普通 full prefill 不受影响；chunked/cache prefill 可能输出错误，稀疏 pattern 也可能异常。
- **修复建议**：v1 若只支持 full prefill + q_len=1 decode，应显式断言 `S_q == S_k`；若要支持 chunked prefill，所有 mask 与 slash index 都统一使用 `q_offset = S_k - S_q`。
- **状态**：`[ ]`

### 12.2 重大性能 / OOM 风险

#### P5 — Vertical-slash CPU Python index convert 仍是主路径性能大坑
- **文件**：`minference/ops/vertical_slash_kernel_npu.py:234-237`、`minference/backend_npu/cuda_shim.py:201-231`
- **根因**：每次 vertical/slash attention 都将 `v_idx/s_idx` 从 NPU 拷到 CPU 并 `.tolist()`，然后在 `B × H × num_rows` 三层 Python 循环中运行双指针扫描，循环内还创建小 tensor 写回。
- **规模估算**：默认 `NNZ_V≈1000`、`NNZ_S≈6096`。128k 序列 `num_rows≈2000`，32 heads 下是 `32 × 2000 × ~7000` 级 Python 操作，且包含 NPU→CPU 同步。
- **后果**：长上下文 prefill 可能被 CPU Python convert 主导，明显偏离 MInference 加速目标。
- **修复建议**：短期用 NumPy/C++/Cython 向量化并减少循环内 tensor 分配；中期回到原方案，用 Triton-Ascend/AscendC 实现 index convert。
- **状态**：`[ ]`

#### P6 — M3/M4 展开 token 级 `[B,H,S_q,S_k]` bool mask，16k 已很重
- **文件**：`minference/ops/vertical_slash_kernel_npu.py:109-159`、`minference/ops/block_sparse_kernel_npu.py:145-165`
- **根因**：block-sparse 与 vertical/slash 当前都走 `npu_fusion_attention(sparse_mode=1, atten_mask=mask)`，mask 是 token 级二维展开，而不是直接消费 block/index 稀疏描述。
- **规模估算**：单 head `S=16384` 的 bool mask 约 256MiB；32 heads 若批量化则约 8GiB，当前 per-head 循环也会反复构建/释放大 mask。block-sparse 的 `expand().reshape()` 可能物化大张量。
- **后果**：8k 以内可用于 correctness smoke；16k 性能和内存压力已大；32k/128k 不适合性能验收。
- **修复建议**：M3/M4 性能路径应改为真正的块/索引稀疏 kernel（TileLang/Triton-Ascend/AscendC），避免 token mask 展开。
- **状态**：`[ ]`

#### P7 — 超过 16384 后 silent dense fallback，长上下文性能测试会失真
- **文件**：`minference/ops/vertical_slash_kernel_npu.py:324-335`、`minference/ops/block_sparse_kernel_npu.py:308-319`
- **根因**：`S > _MAX_SEQ_FOR_MASK` 时发 warning 后直接退化为 `dense_attention()`。vertical/slash 路径还会先在 `minference_forward.py:217-238` 完成 QK/topk 在线估计，再 fallback dense。
- **触发条件**：32k/128k 长上下文，这是迁移目标场景。
- **后果**：实际复杂度退化为 O(S²) dense，128k 基本不可用；M5 若未检查 warning，会误以为在测 sparse kernel 性能。
- **修复建议**：对 `attn_type="minference"` 的性能验收路径，超过阈值不应 silent dense fallback；要么明确报错，要么要求启用真正 sparse kernel。smoke-only fallback 可保留但必须在报告中标注。
- **状态**：`[ ]`

#### P8 — Prefill per-head 循环造成大量 NPU kernel launch
- **文件**：`minference/modules/minference_forward.py:368-382`
- **根因**：每层每 head 单独切片并调用 attention kernel；streaming head 至少两次 `npu_fusion_attention`，M3/M4 还会每 head 构建 mask。
- **规模估算**：32 layers × 32 heads 可达上千次小 kernel launch；NPU launch 固定开销会被放大。
- **修复建议**：按 pattern 类型与参数分组，把相同 pattern 的 heads 批量传入 NPU op；dense 与 streaming 分支可以优先合并。
- **状态**：`[ ]`

#### P9 — GQA/MQA 的 `repeat_kv` 放大 KV 与后续 mask/attention 工作量
- **文件**：`minference/modules/minference_forward.py:351-354`
- **根因**：`repeat_kv()` 将 KV heads 物理复制到 Q heads 数量。对 GQA/MQA 模型，KV 带宽、mask 构建和 attention 工作量都会按 `num_key_value_groups` 放大。
- **触发条件**：Qwen/Llama 等 GQA 模型。
- **后果**：correctness 路径可接受，但性能路径会显著偏慢、占用更多 HBM。
- **修复建议**：性能阶段让 NPU op 支持 KV heads 少于 Q heads，kernel 内按 group 映射，避免物理 repeat。
- **状态**：`[ ]`

### 12.3 建议修复顺序

| 优先级 | 编号 | 估计改动量 | 说明 |
|---|---|---|---|
| P0 | B6 | 中 | 不修可能端到端首次 forward 直接挂；先兼容新版 transformers/RoPE |
| P0 | B7 | 小 | dense smoke 语义错误；会干扰 M5 基础链路验证 |
| P1 | B8 | 小 | torch_npu 返回格式差异会导致 streaming NPU 崩溃 |
| P1 | B9 | 小-中 | 若 v1 不支持 chunked prefill，先显式断言；否则统一 offset |
| P2 | P7 | 小 | 性能测试前必须避免 silent dense fallback 误导结论 |
| P2 | P5 | 大 | vertical/slash 长上下文性能瓶颈，需向量化或 NPU kernel |
| P2 | P6 | 大 | token 级 mask 限制 M3/M4 性能上限 |
| P3 | P8 | 中-大 | head 分组 batching 降低 launch 开销 |
| P3 | P9 | 中-大 | GQA/MQA 性能优化，避免物理 repeat KV |
