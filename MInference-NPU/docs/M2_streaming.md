# M2 — Streaming (A-shape) Attention NPU 实现

> 状态：**已完成**（2026-05-23）
> 对应文件：`minference/ops/streaming_kernel_npu.py`

---

## 1. 算法背景

**Streaming-LLM（A-shape）** 是 MInference 1.0 三种稀疏模式之一（`stream_llm`）。
对于 prefill 阶段第 `i` 个 query，其可见的 key 集合为：

```
visible(i) = sink ∪ sliding
           = { K[j] : j < n_init }           ← 前 n_init 个 token（"sink"）
           ∪ { K[j] : i_abs - n_local < j ≤ i_abs }  ← 宽 n_local 的滑动窗口
```

其中 `i_abs = k_len - q_len + i` 为 query 的绝对位置（处理 KV-cache 场景时与 prefill-only 无异）。

默认配置（`Llama-3.1-8B-Instruct-128k`）：`n_init = 128`，`n_local = 3968`。

---

## 2. NPU 实现路径

### 2.1 为什么用两段而不是一段

`npu_fusion_attention` 的 `sparse_mode=4`（band）能表达 sliding window，但它的
`pre_tockens` / `next_tockens` 参数定义的是以 **当前 query 为中心** 的 band，无法同时
表达"前 n_init 个 token 永远可见"的 sink 段。`prefix` 参数在不同 torch_npu 版本语义不
稳定，不作为 v1 依赖。

因此采用 **两段 + log-sum-exp 合并**：

| 段 | 内容 | `npu_fusion_attention` 参数 |
|---|---|---|
| 段 1 (sliding) | 仅滑动窗口，全 K | `sparse_mode=4, pre_tockens=n_local-1, next_tockens=0` |
| 段 2 (sink) | 仅前 `n_init` 个 key，mask 掉 sliding 重叠 | `sparse_mode=1`（user mask），`k = k[:n_init]` |

### 2.2 两段 LSE 合并公式

设段 1 返回 `(o₁, m₁, l₁)`，段 2 返回 `(o₂, m₂, l₂)`，其中：
- `m = max(scaled logits)`（per-query 最大值，shape `[B, H, S, 1]`）
- `l = Σ exp(scaled logits - m)`（归一化分母，同 shape）
- `o = Σ exp(scaled logits - m) · v`（未归一化输出，shape `[B, H, S, D]`）

合并：

```
m_new = max(m₁, m₂)
α₁    = l₁ · exp(m₁ − m_new)
α₂    = l₂ · exp(m₂ − m_new)
denom = α₁ + α₂
o_new = (α₁ · o₁ + α₂ · o₂) / denom
```

当段 2 整张 mask 全部为 True（所有 early query 无 sink 贡献）时 `l₂ = 0`，直接返回
`o₁` 跳过合并，避免 `-inf - (-inf) = NaN`。

### 2.3 短路逻辑

| 条件 | 处理 |
|---|---|
| `k_len <= n_local` | 所有 K 在 sliding window 内 → 退化 causal dense |
| `n_init <= 0` | 无 sink → 跑段 1 后直接返回 `o₁` |

---

## 3. 接口约定

```python
from minference.ops.streaming_kernel_npu import streaming_forward

out = streaming_forward(q, k, v, n_init=128, n_local=3968)
```

| 参数 | 形状 | 约定 |
|---|---|---|
| `q, k, v` | `[B, H, S, D]` (BNSD) | 已 RoPE、已 `repeat_kv`（head 数对齐） |
| `n_init` | int | sink 段长度，`<= 0` 表示无 sink |
| `n_local` | int | sliding window 宽度 |
| 返回 | `[B, H, S, D]` | 同 dtype、同 device |

**head_dim pad**：若 `D ∉ {16,32,64,128,256,512}`，函数自动 pad 到下一个允许值，输出截回
原始 `D`，对调用方透明。

**device 路由**：`q.device.type == "npu"` 且 `torch_npu` 可用 → NPU 两段路径；否则 →
PyTorch ref（CPU / CUDA 兜底）。

---

## 4. 调用链（M2 完成后）

```
minference/__init__.py
    └─ from .ops.streaming_kernel_npu import streaming_forward   ← M2 接入

minference/modules/minference_forward.py
    gather_last_q_vertical_slash_topk_v4(...)
        if ty == "stream_llm":
            return _streaming_forward(q, k, v, n_init=vertical_size, n_local=slash_size)
        # block_sparse / vertical_and_slash → dense（M3/M4 时替换）
```

上游约定（`MInference/minference/modules/minference_forward.py:481-486`）：
`stream_llm` 分支的 `best_pattern` 条目把 `(vertical_size, slash_size)` 复用为
`(n_init, n_local)`。

---

## 5. 测试跑法

```bash
# CPU/非 NPU 环境（ref 正确性 + 短路 + head_dim pad）
cd MInference-NPU
python -m pytest tests/test_streaming_kernel.py -v

# NPU 环境（额外跑 test_npu_vs_pytorch_ref）
python -m pytest tests/test_streaming_kernel.py -v
```

测试文件：`tests/test_streaming_kernel.py`

| 测试组 | 描述 | 容差 |
|---|---|---|
| `test_pytorch_ref_vs_naive` | `_streaming_pytorch_ref` vs 逐 row naive | < 1e-3 |
| `test_streaming_forward_vs_naive` | 顶层 `streaming_forward` vs naive（含短路） | < 1e-2（fp16）/ 1e-3（fp32） |
| `test_head_dim_pad_roundtrip` | 非标 head_dim 进出形状一致 | — |
| `test_npu_vs_pytorch_ref` | NPU `_streaming_npu` vs PyTorch ref | < 1e-2 |
| `test_param_sweep` | 多组 `(n_init, n_local)` 参数边界扫描 | < 1e-3 |

---

## 6. 与 M3 的衔接

M3 目标：把 `block_sparse` 分支从 `dense_attention` 替换为真正的 NPU block-sparse kernel。
入口同样在 `gather_last_q_vertical_slash_topk_v4`，按 `ty == "block_sparse"` 分支替换。
NPU 实现候选：`npu_fusion_attention sparse_mode + atten_mask`，或 TileLang
`examples/blocksparse_attention/`。

---

## 7. 与上游差异对照

| 项目 | 上游（CUDA/Triton） | NPU v1 |
|---|---|---|
| 实现文件 | `ops/streaming_kernel.py` | `ops/streaming_kernel_npu.py` |
| sliding window | 自定义 Triton kernel（custom mask） | `npu_fusion_attention(sparse_mode=4)` |
| sink 段 | 同一 Triton kernel（mask 拼接） | 单独第二段 FA + LSE 合并 |
| head_dim pad | 上游同样有 | 相同逻辑，复用 `_ALLOWED_HEAD_DIMS` |
| CPU 兜底 | 无（CUDA-only） | `_streaming_pytorch_ref`（分块 fp32） |
| 短路（k_len≤n_local） | 直接 causal dense flash_attn | `backend_npu.dense_attention` |
