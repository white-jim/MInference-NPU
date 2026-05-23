# M3 — Block-Sparse kernel NPU 实现

> 本文档描述 M3 milestone 的实现细节，对应 `docs/migration_plan_v1.md` §4 里的 M3 节点。
> 假设读者已了解 M0/M1/M2，可对照 `docs/M2_streaming.md` 阅读。

---

## 1. 算法背景

**Block-Sparse Attention** 是 MInference 三种稀疏模式之一，`best_pattern` 中
`ty == "block_sparse"` 的 head 使用此模式。与 `stream_llm`（A-shape 窗口）和
`vertical_and_slash`（散列稀疏 + 反对角线）不同，block-sparse 按固定粒度（block_size=64）
把 KV 序列切块，再用 block 级别的 top-k 选择最重要的 key blocks。

### 1.1 上游实现（CUDA）

上游 `MInference/minference/ops/block_sparse_flash_attention.py` 使用：
1. `_build_block_index`：mean-pool Q/K → block 级 QK 得分 → top-k → block index 表（整数）
2. Triton kernel `_triton_block_sparse_attn_fwd_kernel`：按 block index 表 gather K/V，
   在 kernel 内做 online softmax（与 vertical_and_slash 类似）

### 1.2 NPU v1 实现（本 milestone，路径 A）

v1 选择**路径 A**：把 block-sparse 模式转换成 bool mask，然后用
`npu_fusion_attention(sparse_mode=1, atten_mask=mask)` 实现。

优点：无需 Triton-Ascend kernel，API 稳定，实现最简。
缺点：mask 构建是 O(S_q × S_k)，序列长度受限（≤ 16384）。超长序列见 §6 路径 B。

---

## 2. mask 构建详解

`_build_block_sparse_mask(q, k, topk_blocks, block_size)` 分 6 步：

```
1. pad Q/K 到 block_size 的整数倍（pad_q = -S_q % block_size）
2. mean-pool → q_pool [B, H, n_bq, D]，k_pool [B, H, n_bk, D]
3. block 级 QK 打分：scores = q_pool @ k_pool.T * scale  [B, H, n_bq, n_bk]
4. 施加 block 级因果约束：scores[bq, bk] = -inf  if bq < bk
5. 每个 query block 选 top-k key blocks（by score）
6. 展开：block_attend [B,H,n_bq,n_bk] → token_attend [B,H,S_q,S_k]
   + 追加 per-token 因果约束（j <= abs_i，abs_i = S_k-S_q+i）
```

**为什么要 per-token 因果约束？**
Block 级 top-k 只保证 query block `bq` 不选 `bk > bq` 的 key block。但在对角
block（`bk == bq`）内，query token `i`（bq block 的首几个 token）不应看到 key token
`j > i`（同一 block 内靠后的 token）。per-token 因果约束消除这一问题，是正确性的
最终保障。

**reshape 合法性验证（不需要 permute）：**

```
block_attend [B,H,n_bq,n_bk]
→ .unsqueeze(3).unsqueeze(5)           [B,H,n_bq,1,n_bk,1]
→ .expand(-1,-1,-1,bs,-1,bs)          [B,H,n_bq,bs,n_bk,bs]
→ .reshape(B,H,n_bq*bs,n_bk*bs)       [B,H,S_q_p,S_k_p]
```

flat_index[bq,bqi,bk,bki] = bq*(bs*n_bk*bs) + bqi*(n_bk*bs) + bk*bs + bki
= (bq*bs + bqi) * (n_bk*bs) + (bk*bs + bki)
= i * S_k_p + j  ✓

---

## 3. NPU 实现路径

```python
# block_sparse_kernel_npu.py

mask = _build_block_sparse_mask(q, k, topk_blocks, block_size)  # [B,H,S_q,S_k] bool

result = torch_npu.npu_fusion_attention(
    q, k, v,
    head_num=n_heads,
    input_layout="BNSD",
    scale=head_d**-0.5,
    sparse_mode=1,       # user-provided mask
    atten_mask=mask,     # True = masked out
)
out = result[0]
```

**mask 约定**（与 M2 一致）：`True` = 被遮蔽（不参与 attention），即 CANN
`npu_fusion_attention` 的标准惯例。

---

## 4. 接口约定

### 4.1 顶层函数

```python
from minference.ops.block_sparse_kernel_npu import block_sparse_attention

out = block_sparse_attention(
    q,              # [B, H, S, D]
    k,              # [B, H, S, D]
    v,              # [B, H, S, D]
    topk_blocks,    # int：保留的 key block 数（对应上游 top_k 参数）
    block_size=64,  # int：block 大小（默认 64，与上游一致）
)  # -> [B, H, S, D]
```

`topk_blocks` 来自 `best_pattern` JSON 的 `vertical_size` 字段（block_sparse 模式下
的语义是"top-k block 数"）。

### 4.2 模块门面（`minference/__init__.py`）

```python
from minference import block_sparse_attention  # 已从 block_sparse_kernel_npu 导出
```

### 4.3 per-head 调度器接入点（`minference_forward.py`）

```python
# gather_last_q_vertical_slash_topk_v4，block_sparse 分支（M3 已替换）：
if ty == "block_sparse":
    return _block_sparse_attention(q, k, v, topk_blocks=int(vertical_size))
```

---

## 5. 调用链

```
minference_forward.py:gather_last_q_vertical_slash_topk_v4
  └─ ty == "block_sparse"
       └─ block_sparse_kernel_npu.block_sparse_attention(q, k, v, topk_blocks=vertical_size)
            ├─ [NPU]  _block_sparse_npu → _build_block_sparse_mask → npu_fusion_attention
            └─ [CPU]  _block_sparse_pytorch_ref → _build_block_sparse_mask → masked softmax
```

---

## 6. 序列长度限制与路径 B

| 序列长度 | mask 大小（1×1×S×S bool） | 建议 |
|---|---|---|
| ≤ 8192   | ≤ 64 MB  | 路径 A ✓ |
| ≤ 16384  | ≤ 256 MB | 路径 A 可用（`_MAX_SEQ_FOR_MASK` 默认上限）|
| > 16384  | > 256 MB | **路径 A 自动退化为 dense + WARNING** |

**路径 B**（v2 计划）：用 TileLang-Ascend `examples/blocksparse_attention/` 或自研
Triton-Ascend kernel 实现 block-sparse，只访问选中的 K/V block，内存 O(topk × block_size × D)。
详见 `docs/ascend_migration_survey.md` Block-Sparse 行。

---

## 7. 测试跑法

```bash
# CPU / 非 NPU 环境（跑所有非 skip 测试）：
python -m pytest tests/test_block_sparse_kernel.py -v

# NPU 环境（额外跑 NPU vs ref 对比）：
python -m pytest tests/test_block_sparse_kernel.py -v

# 只跑某一组：
python -m pytest tests/test_block_sparse_kernel.py::test_pytorch_ref_full_topk_vs_dense -v
python -m pytest tests/test_block_sparse_kernel.py::test_npu_vs_pytorch_ref -v  # NPU 专属
```

---

## 8. 与上游差异对照

| 维度 | 上游（CUDA Triton）| NPU v1（本文档）|
|---|---|---|
| kernel 类型 | Triton `@jit` kernel | npu_fusion_attention + bool mask |
| mask 表达 | block index 整数表 `[B,H,n_bq,topk]` | token 级 bool mask `[B,H,S_q,S_k]` |
| 序列长度 | 无限制（Triton 按需 gather） | ≤ 16384（mask O(S²)），超出退化为 dense |
| head_dim 约束 | `{16,32,64,128}`（Triton constexpr） | `{16,32,64,128,256,512}`（自动 pad）|
| causal 保证 | kernel 内逐 token 检查 `cols <= offs_m` | per-token bool 遮罩 + NPU mask |
| block_size 参数 | `block_size_M`, `block_size_N`（可各自设置）| `block_size`（M == N，v1 简化）|

---

## 9. 与 M4 衔接

M3 完成后，`minference_forward.py` 中只剩 `vertical_and_slash` 分支仍退化为 dense。
M4 分两步：

- **M4-a**：实现 `backend_npu/cuda_shim.py:convert_vertical_slash_indexes`（NPU 版 CUDA
  索引展开 kernel）。
- **M4-b**：实现 `ops/pit_sparse_flash_attention_v2.py` NPU vertical-slash kernel（Triton-Ascend），
  接入 `minference_forward.py` 的 `vertical_and_slash` 分支。

M4-b 是 v1 最复杂的部分，参见 `docs/ascend_migration_survey.md` Vertical-Slash 行。
