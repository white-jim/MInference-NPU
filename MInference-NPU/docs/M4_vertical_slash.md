# M4 — Vertical-Slash Sparse Attention（NPU v1）

## 1. 算法背景

Vertical-and-slash 是 MInference 1.0 中最主要的稀疏模式（约 95% 的 head 使用）。其注意力
模式由两个稀疏成分叠加而成：

| 成分 | 形态 | 含义 |
|---|---|---|
| **Vertical（竖向列）** | 若干列完整可见 | 始终对整段序列重要的位置（sink token + 高频 topk 列） |
| **Slash（斜线段）** | 反对角线带状区间 | 相对位置固定的局部 pattern（sliding window 的泛化） |

两者叠加后，每个 query token 只 attend 有限数量的 KV token（远少于全序列），实现稀疏加速。

### 1.1 在线估计流程

每个 prefill step：
1. 取最后 `last_q = min(64, S)` 个 query token 与全 K 做一次 QK 乘法（近似 attention 权重）。
2. 列和 → top-k → `v_idx`（vertical 列，升序），强制保留前 30 个 sink 列。
3. 反对角线和（`sum_all_diagonal_matrix`）→ top-k → `s_idx`（slash 偏移，降序），
   强制保留最近 100 条 local slash。

---

## 2. M4-a：`convert_vertical_slash_indexes`（CPU Python）

### 2.1 对应关系

| 上游 | NPU v1 |
|---|---|
| `csrc/vertical_slash_index.cu` CUDA kernel | `backend_npu/cuda_shim.py` Python 实现 |
| 输入 `vertical_indexes`（升序）+ `slash_indexes`（降序） | 相同签名 |
| 输出 `(block_count, block_offset, column_count, column_index)` | 相同四元组，CPU int32 tensor |

### 2.2 算法（双指针，与 CUDA 完全等价）

对每个 `(batch, head, query_block [start_m, end_m))`：

```
跳过 slash_indexes[s] >= end_m 的无效项
初始化 range_end = max(end_m - s_raw, BLK)
       range_start = range_end - BLK

双指针主循环：
  if v_val < range_end:           ← vertical 值落在当前 range 之前
    if v_val < range_start:       ← 不在 slash 覆盖范围 → 加入 column_index
      col_buf.append(v_val)
    advance v
  else:                           ← v 已超过 range_end → 推进 slash
    new_range_end = max(end_m - slash_indexes[s++], BLK)
    if new_range_end > range_end + BLK:
      save current range → blk_buf   # 有间隔，开新 range
      range_start = new_range_end - BLK; range_end = new_range_end
    elif new_range_end > range_end:
      range_end += BLK               # 相邻/重叠，扩展当前 range
    else:
      pass                           # 已覆盖，跳过
  if s exhausted: save current range; break
```

**关键属性**：
- Slash 相邻段合并为连续 KV 块（减少 kernel 调用次数）。
- Vertical 列去重（落在 slash range 内不重复计入 column_index）。
- 双指针 O(NNZ_V + NNZ_S) per query block，整体 O((NNZ_V + NNZ_S) × num_rows)。

### 2.3 设计决策：CPU 侧 Python 实现

NNZ_V ≈ 1000，NNZ_S ≈ 6096，num_rows = S/64。对于 S = 32k：512 query blocks，每个 block
运行双指针约 7000 次比较。单 prefill pass 约 3.5M Python 操作，无 GPU 同步等待，CPU 侧
<1ms，不构成瓶颈。

Triton-Ascend 重写（方案 B）仅在 CPU 方案成为实测吞吐瓶颈时考虑。

---

## 3. M4-b：Vertical-Slash NPU Kernel（路径 A）

### 3.1 整体流程

```
v_idx, s_idx (在线估计)
      │
      ▼ convert_vertical_slash_indexes (CPU, M4-a)
(block_count, block_offset, column_count, column_index)
      │
      ▼ _build_vs_mask_from_indexes
token 级 bool mask [B, H, S_q, S_k]  (True = masked)
      │
      ▼ npu_fusion_attention(sparse_mode=1, atten_mask=mask)
attn_output [B, H, S_q, D]
```

### 3.2 mask 构建：cumsum 区间标记法

对每个 query block `bq`（行范围 `[start_q, end_q)`）：

```python
# Slash 覆盖：用 cumsum 区间标记，O(NNZ_S + S_k) 而非 O(NNZ_S × S_k)
cov_delta = torch.zeros(S_k + 1)
cov_delta.scatter_add_(0, blk_starts, +1)
cov_delta.scatter_add_(0, blk_ends,   -1)
slash_cov = cov_delta.cumsum(0)[:S_k] > 0  # [S_k] bool

# Vertical 覆盖：scatter set
vert_cov[column_index] = True

# 合并 + 因果约束
combined = slash_cov | vert_cov
attend_2d = combined[None, :] & (j_idx <= q_rows[:, None])  # [block_rows, S_k]
mask[start_q:end_q, :] &= ~attend_2d
```

### 3.3 序列长度限制

| max(S_q, S_k) | 行为 |
|---|---|
| ≤ 16384 | NPU 路径（路径 A） |
| > 16384 | 发 WARNING，退化为 causal dense |

路径 B（Triton-Ascend 直接使用 block_offset/column_index 的稀疏 kernel）可突破此限制。

### 3.4 head_dim pad

与 M2/M3 一致：`head_dim ∉ {16,32,64,128,256,512}` 时自动 pad 到 2 的幂，输出截回原始
head_dim。

---

## 4. 接口约定

```python
from minference.ops.vertical_slash_kernel_npu import vertical_slash_sparse_attention

out = vertical_slash_sparse_attention(
    query,           # [B, H, S_q, D]  BNSD，已 RoPE，已 repeat_kv
    key,             # [B, H, S_k, D]
    value,           # [B, H, S_k, D]
    v_idx,           # [B, H, NNZ_V]  int32 升序
    s_idx,           # [B, H, NNZ_S]  int32 降序
    block_size_M=64,
    block_size_N=64,
)  # -> [B, H, S_q, D]
```

---

## 5. 调用链（M4 完整路径）

```
LlamaAttention.forward (monkey-patched by patch.py)
  └─ minference_forward (minference_forward.py)
       └─ gather_last_q_vertical_slash_topk_v4  # per-head 调度
            ├─ ty == "vertical_and_slash"
            │   └─ _vertical_and_slash_kernel   # 在线估计 v_idx/s_idx
            │        └─ _vs_sparse_attention    # vertical_slash_sparse_attention alias
            │             ├─ convert_vertical_slash_indexes  (CPU, M4-a)
            │             ├─ _build_vs_mask_from_indexes     (CPU → NPU tensor)
            │             └─ npu_fusion_attention(sparse_mode=1)
            ├─ ty == "block_sparse"  → _block_sparse_attention  (M3)
            └─ ty == "stream_llm"   → _streaming_forward        (M2)
```

---

## 6. 测试跑法

```bash
# CPU 环境（所有组：T1-T6、T8；T7 自动 skip）
python -m pytest tests/test_vertical_slash_kernel.py -v

# NPU 环境（额外跑 T7）
python -m pytest tests/test_vertical_slash_kernel.py -v

# 仅跑 convert_vertical_slash_indexes 测试
python -m pytest tests/test_vertical_slash_kernel.py -v -k "Convert"

# 仅跑 NPU 测试
python -m pytest tests/test_vertical_slash_kernel.py -v -k "npu"
```

CPU 容差：`atol = 1e-3`（fp32）/ `5e-2`（fp16）。
NPU 容差：`atol = 1e-2`。

---

## 7. 与上游差异对照

| 上游 | NPU v1（M4） |
|---|---|
| 调用 `minference.cuda.convert_vertical_slash_indexes`（CUDA C++） | 调用 `backend_npu.cuda_shim.convert_vertical_slash_indexes`（CPU Python） |
| `_triton_mixed_sparse_attn_fwd_kernel`（CUDA/Triton kernel，直接消费索引） | `npu_fusion_attention(sparse_mode=1)`（消费 token 级 bool mask） |
| 支持任意序列长度（稀疏 Triton kernel） | 路径 A 限制 ≤ 16384（O(S²) mask）；路径 B 可突破（v2 计划） |
| `block_size_M/N = 64`（硬编码） | 同 64（路径 A 限制） |
| sgl_kernel 优化路径（`convert_vertical_slash_indexes_opt`） | 不适用 |

---

## 8. 与 M5 衔接

M4 完成后，所有三种稀疏分支（vertical_and_slash / block_sparse / stream_llm）均已接入真
NPU kernel。M5 进行端到端联调：

1. `python examples/run_hf_minimal.py --attn_type minference --model Llama-3.1-8B ...`
2. 对比 GPU（上游）和 NPU 的 perplexity / RULER 得分。
3. 性能基准：prefill 时延 vs dense baseline。
4. 实机 checklist：`docs/SETUP.md §6`。
