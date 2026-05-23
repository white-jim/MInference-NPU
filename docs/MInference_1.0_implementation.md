# MInference 1.0 实现详解

> 目标读者：希望深入理解 MInference 1.0 算法实现、kernel 设计与代码组织的开发者 / 研究者。
> 参考论文：*MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention*（NeurIPS'24 spotlight, arXiv:2407.02490）。

---

## 0. 一句话概括

MInference 1.0 利用长上下文 LLM **attention 的动态稀疏性**对 prefill 阶段做加速：
离线为每个 head 选定一个**静态稀疏模式**（A-shape / Vertical-Slash / Block-Sparse），
在线用最后 64 个 query 估出**当前稀疏索引**，再用专门的 Triton + CUDA kernel 跳过被剪掉的 K/V 块，
在单卡 A100 上把 1M token 的 prefill 加速最多 **10×**，精度几乎无损。

---

## 1. 总体架构

```
┌────────────────────────────── 用户入口 ──────────────────────────────┐
│  MInference(attn_type="minference", model_name=..., kv_type=...)    │
│        ↓ patch_model()                                                │
│  monkey-patch HF / vLLM 的 LlamaAttention.forward                    │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────── minference_forward (per layer) ───────────────────┐
│  按 head 循环：                                                       │
│    1) 读 best_pattern[layer][head] = (ty, v_size, s_size, score)     │
│    2) 在线估计稀疏索引（仅末 64 个 query 参与）                       │
│    3) 调对应 kernel:                                                  │
│       - vertical_and_slash → vertical_slash_sparse_attention (主力)  │
│       - block_sparse       → block_sparse_attention                  │
│       - stream_llm         → streaming_forward (A-shape)             │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────── 算子层：Triton kernel + CUDA index ──────────────────┐
│  CUDA: convert_vertical_slash_indexes                                │
│        (列下标 + 对角线下标 → block_count/offset + column_count/idx) │
│  Triton: _triton_mixed_sparse_attn_fwd_kernel (Vertical-Slash FA)    │
│  Triton: _triton_block_sparse_attn_fwd_kernel (Block-Sparse FA)      │
│  Triton: streaming_kernel (Sink + Sliding Window FA)                 │
│  优先调用 sglang / vllm 的 sparse_attn_func（同语义、更快）           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. 代码组织（仅列出 MInference 1.0 相关）

```
MInference/
├── csrc/                                  ★ CUDA 源码（通过 setup.py 编译为 minference.cuda）
│   ├── kernels.cpp                        ─ pybind 入口，导出 convert_vertical_slash_indexes
│   └── vertical_slash_index.cu            ─ 稀疏索引展开 kernel
│
├── minference/
│   ├── __init__.py                        ─ 对外导出三个 op + MInference 类
│   ├── minference_configuration.py        ─ attn_type / kv_type 枚举与校验
│   ├── models_patch.py                    ─ MInference 顶层入口，分发到各 attn_type
│   ├── patch.py                           ─ HF / vLLM 的实际 monkey-patch 逻辑
│   ├── utils.py                           ─ 通用工具（RoPE、prepare_input …）
│   │
│   ├── modules/
│   │   ├── minference_forward.py          ★ MInference 1.0 的核心 forward（per-head 调度）
│   │   ├── forward.py                     ─ 通用 attn_forward 调度器（含 prefill_forwards 字典）
│   │   └── kvcompression.py               ─ 多种 KV 压缩方法（与 attn 解耦，可正交组合）
│   │
│   ├── ops/
│   │   ├── pit_sparse_flash_attention_v2.py   ★ Vertical-Slash 主 Triton kernel
│   │   ├── pit_sparse_flash_attention.py       ─ v1（早期）
│   │   ├── pit_sparse_flash_attention_v3.py    ─ v3（FA3 / Hopper 优化）
│   │   ├── block_sparse_flash_attention.py    ★ Block-Sparse Triton kernel
│   │   ├── streaming_kernel.py                ★ A-shape (sink + sliding window) Triton kernel
│   │   └── flash_attn_triton.py               ─ Triton 版 FlashAttention（无 flash-attn 时 fallback）
│   │
│   ├── configs/                           ★ 离线搜得的 best_pattern 配置
│   │   ├── model2path.py                  ─ model_name → 配置文件路径
│   │   ├── Llama_3.1_8B_Instruct_128k_*.json
│   │   ├── Qwen2.5_7B_Instruct_1M.json
│   │   ├── GLM_4_9B_1M_instruct_*.json
│   │   └── ...
│   │
│   └── dist_ops/                          ─ 多卡 attention（Ring/Striped/Zigzag）+ MInference
│
├── examples/
│   ├── run_hf.py / run_vllm.py            ─ 最小调用示例
│   └── run_hf_streaming.py / .sh
│
└── experiments/                           ─ 评测脚本
    ├── benchmarks/                        ─ E2E latency / TP 测试
    ├── needle_in_a_haystack/, ruler/, infinite_bench/, ppl/
```

---

## 3. 算法层：三种稀疏模式

每个 head 在 offline 搜索阶段被打上一个 `(pattern_type, v_size, s_size, score)` 标签，写入 `configs/*.json`。
示例（Llama-3.1-8B 第 0 层前若干 head）：

```json
[
  {
    "0":  ["vertical_and_slash", 1000, 6096, 0.96],
    "1":  ["vertical_and_slash", 1000, 6096, 0.95],
    ...
    "12": ["vertical_and_slash", 1000, 6096, 0.59],
    ...
  },
  ...
]
```

### 3.1 Vertical-Slash（主力，覆盖绝大多数 head）

注意力图中：少数若干 **垂直列**（某些 key 几乎对所有 query 都有响应）+ 若干 **对角斜线**（与某些相对位置距离的 key 强响应）。

- `v_size`：保留的 vertical 列数（默认 1000）
- `s_size`：保留的 slash 对角线数（默认 6096）

### 3.2 Block-Sparse

注意力图近似按 64×64 block 稀疏，每个 query block 取 top-k 个 K block。
默认 `top_k = 100`（见 `block_sparse_kernel`）。

### 3.3 A-shape / Stream-LLM

固定保留 sink（前 `n_init` 个 token）+ sliding window（最近 `n_local` 个 token），是 `vertical_and_slash` 的退化情形。
在 `models_patch.py` 中默认 `n_init=128`, `n_local=3968`。

> 注：`gather_last_q_vertical_slash_topk_v4` 中即使 best_pattern 是 `stream_llm`，最终仍调用 `vertical_and_slash_kernel`，因为后者参数化退化即可。`search_pattern` (L222) 显式做了 `stream_llm → vertical_and_slash` 的替换。

### 3.4 静态/对照模式

- `static`：第一次估出的 vertical/slash 索引被缓存到 `self.vs` 复用，后续 forward 完全静态。
- `dilated1` / `dilated2`：消融实验用的固定 dilated 模式。
- `dense`：直接走 flash_attn（基线）。

---

## 4. Python 层：核心 forward 流程

文件：`minference/modules/minference_forward.py`（1361 行）

### 4.1 入口 `minference_forward()`（L497）

返回一个被 monkey-patch 进 `LlamaAttention.forward` 的闭包：

```python
def forward(self, hidden_states, attention_mask, position_ids,
            past_key_value, output_attentions, use_cache, **kwargs):
    self.init_minference_parameters()       # 加载本层 best_pattern
    bsz, q_len, _ = hidden_states.size()

    # 1. QKV projection（兼容 Phi 的 qkv 合一与 Llama 的分离）
    query, key, value = ...

    # 2. RoPE（按 set_rope_type / get_cos_sin 兼容多家模型）
    query_states, key_states = apply_rotary_pos_emb(...)

    # 3. 更新 KV cache
    if past_key_value is not None:
        key_states, value_states = past_key_value.update(...)

    # 4. repeat_kv (GQA)
    key_states  = repeat_kv(key_states,  num_key_value_groups)
    value_states= repeat_kv(value_states,num_key_value_groups)

    # 5. prefill 走 per-head 稀疏；decoding (q_len==1) 走 dense flash_attn
    if q_len != 1:
        output = torch.empty_like(query_states)
        for head in range(query_states.size(1)):
            q = query_states[:, head:head+1]
            k = key_states  [:, head:head+1]
            v = value_states[:, head:head+1]
            if layer_idx >= starting_layer:
                attn_output = self.gather_last_q_vertical_slash_topk_v4(q, k, v, head)
            else:
                attn_output = flash_attn_func(...)
            output[:, head:head+1] = attn_output
    else:
        output = flash_attn_func(query_states, key_states, value_states, causal=False)

    return self.o_proj(output.transpose(1,2).reshape(...)), None, past_key_value
```

要点：
- **per-head 循环** —— 不同 head 选不同 pattern，不能 batch 起来。这是论文公开实现的写法，对开销不敏感是因为稀疏 kernel 本身节省的时间远多于循环。
- **starting_layer** —— 前若干层经验上不适合稀疏（attention 还没收敛成静态 pattern），保持 dense flash_attn。
- **decoding** —— MInference 1.0 只加速 prefill；解码阶段单 query，直接 dense。

### 4.2 单 head 调度 `gather_last_q_vertical_slash_topk_v4`（L296）

```python
def gather_last_q_vertical_slash_topk_v4(self, q, k, v, head_id):
    q_len = q.shape[2]
    if q_len == 1:
        return dense(q, k, v)

    # 各种 override（dense / a_shape / tri_shape / dilated / static / vs_only ...）
    ...

    ty, vertical_size, slash_size, _ = self.best_pattern.get(
        head_id, ("vertical_and_slash", 1000, 6096, 1))

    fc = {
        "stream_llm":         streaming_forward,
        "vertical_and_slash": vertical_and_slash_kernel,
        "block_sparse":       block_sparse_kernel,
    }[ty]
    return fc(q, k, v, vertical_size, slash_size)
```

### 4.3 在线估计稀疏索引 `vertical_and_slash_kernel`（L381）

```python
def vertical_and_slash_kernel(q, k, v, vertical_size, slash_size):
    vertical_size = min(q_len, max(vertical_size, 30))
    slash_size    = min(q_len, max(slash_size,    50))
    last_q = min(64, q_len)

    # (a) 只用最后 64 个 query 做一次 QK
    qk = torch.einsum('bhmk,bhnk->bhmn',
                      q[:,:,-last_q:,:], k) / sqrt(d)
    # 因果 mask（只 mask 末 64 与末 64 之间）
    qk[:,:,:,-last_q:] = torch.where(LAST_Q_MASK[...,-last_q:,-last_q:],
                                     qk[:,:,:,-last_q:], -inf)
    qk = softmax(qk, dim=-1, dtype=fp32)

    # (b) 列累加 → vertical 重要性
    vertical = qk.sum(-2, keepdim=True)
    vertical[..., :30] = inf                       # 强制保留前 30 列（attention sink）
    vertical_topk = torch.topk(vertical, vertical_size, -1).indices

    # (c) 反对角线累加 → slash 重要性
    slash = sum_all_diagonal_matrix(qk)[..., :-last_q+1]
    slash[..., -100:] = inf                        # 强制保留最近 100 条对角线（local window）
    slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

    # (d) 调 Triton/CUDA kernel
    return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)
```

辅助：

- `sum_all_diagonal_matrix(mat)`（L110）：用 `as_strided` 一次性求 QK 矩阵每条反对角线的和，O(n) 内存复杂度，是 vertical-slash 估计的灵魂操作。
- `LAST_Q_MASK`（L43）：一个 `64×64` 的下三角全局 cache，避免每步重建。
- 强制保留前 30 列 / 最近 100 条 slash 是为了维持 sink + local 的基本能力（也对应论文里 vertical-slash 退化为 A-shape 的边界）。

### 4.4 离线搜索 `search_pattern` / `search_pattern_v2`（L129 / L229）

当 `is_search=True` 时启用：
- `search_pattern` 用"近似稀疏 mask × 真实 attn weights"在被 mask 区域剩余的概率质量作为得分（越大越好），遍历候选 `(v_size, s_size)`。候选集见 L207–L213：
  ```
  stream_llm:         (100, 800)
  vertical_and_slash: (30, 800), (100, 750), (500, 700), (3500, 100)
  block_sparse:       (8, 1)
  ```
- `search_pattern_v2`：用真实 sparse kernel 输出与 dense 输出的差异（`(|ref-score|>5e-3).sum()`）作为得分（越小越好）。
- 最优结果按 `(ty, v_size, s_size, score)` 写入 `config_path`，跨层累积写一个 list。

### 4.5 其他变体

- `vertical_and_slash_kernel_static`（L418）：把第一次估计结果缓存到 `self.vs`，后续直接复用。
- `vertical_and_slash_kernel_extend`（L398）：把估计窗口右移 100 token，用于带额外 prompt suffix 的场景。
- `dialted` / `dense` / `block_sparse_kernel` / `tri_shape_kernel`（L365、L438–L455）。
- `kvcompress_forward`（与 `minference/modules/kvcompression.py` 配合）：在 vertical-slash 索引基础上再做 KV cache 压缩（v4 形式）。
- `minference_kv_cache_cpu_forward`（L676+）：把全量 KV 放 CPU、按层异步搬到 GPU，使 1M token 单 A100 可跑。

### 4.6 patch 入口 `minference/patch.py`

- `new_patch(model, config)`：新版统一入口，会替换 `LlamaAttention.forward` 为通用 `attn_forward`，并把 `prefill_forwards["minference"]` 注册成 `minference_prefill_forward`。
- `minference_patch(model, config)`：老版直接替换 `forward` 为 `minference_forward()` 返回的闭包。
- `minference_patch_vllm` / `minference_patch_vllm_tp` / `minference_patch_vllm_executor`：在 vLLM `PagedAttention` 的 forward 中拦截，把稀疏 kernel 挂到 prefill 路径。TP 模式需要手动把 patch 代码拷到 `vllm/worker/worker.py` 里（README 中有说明）。
- `patch_hf`：兼容多家模型（Llama / Mistral / Qwen2 / MiniCPM / Phi-3 / ChatGLM），替换 `self_attn.forward`、`DecoderLayer.forward`、`model.forward`，并接管 RoPE（`RotaryEmbeddingESM`）以支持超长上下文。

---

## 5. 算子层：Triton kernel 实现

### 5.1 Vertical-Slash kernel
文件：`minference/ops/pit_sparse_flash_attention_v2.py`

#### 5.1.1 顶层 wrapper `vertical_slash_sparse_attention`（L195）

```python
def vertical_slash_sparse_attention(query, key, value, v_idx, s_idx,
                                    block_size_M=64, block_size_N=64):
    # 1) padding 到 block_size_M 的整数倍
    pad = (block_size_M - context_size) & (block_size_M - 1)
    query/key/value = F.pad(..., [0,0,0,pad,...])

    # 2) head_dim 若非 2 的幂，pad 到下一 2 幂
    if head_dim not in [16,32,64,128,256,512]:
        target_dim = 2**ceil(log2(head_dim)) - head_dim
        ... pad ...

    # 3) 排序索引：vertical 升序、slash 降序（CUDA kernel 假定）
    v_idx = v_idx.int().reshape(B,H,-1).sort(-1, ascending=True)[0]
    s_idx = s_idx.int().reshape(B,H,-1).sort(-1, descending=True)[0]

    # 4) CUDA 把索引展开成 (block_count, block_offset, column_count, column_index)
    block_count, block_offset, column_count, column_index = \
        convert_vertical_slash_indexes(seqlens, v_idx, s_idx, context_size,
                                       block_size_M, block_size_N)

    # 5) 优先用 sglang/vllm 的 sparse_attn_func（FlashAttention 实现），否则回退到 Triton
    if sparse_attn_func is not None:
        out = sparse_attn_func(q.t(1,2), k.t(1,2), v.t(1,2),
                               block_count, block_offset,
                               column_count, column_index,
                               causal=True).t(1,2)
    else:
        out = _triton_mixed_sparse_attention(...)

    return out[..., :context_size, :head_dim]
```

新版 `vertical_slash_sparse_attention_wo_pad`（L244）使用 `convert_vertical_slash_indexes_opt`（带 causal 参数），无需 pad，是 SGLang/vLLM 已合并的高速路径。

#### 5.1.2 Triton kernel `_triton_mixed_sparse_attn_fwd_kernel`（L48）

这是 **FlashAttention v2 风格** 的 online-softmax kernel，但 K/V 加载分两段：

```python
@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    block_count,   # [B,H, num_M_blocks]            该 query block 要算多少个 K 块
    block_offset,  # [B,H, num_M_blocks, NNZ_S]     这些 K 块的起始位置（slash 段）
    column_count,  # [B,H, num_M_blocks]            该 query block 要算多少个零散列
    column_index,  # [B,H, num_M_blocks, NNZ_V]     这些列的下标（vertical 列）
    Out,
    ...stride...,
    BLOCK_M, BLOCK_N, BLOCK_DMODEL,
):
    start_m = tl.program_id(0)        # query block 索引
    off_hz  = tl.program_id(1)        # batch × head

    # 1) 加载 Q[start_m] 进 SRAM
    q = tl.load(q_ptrs)
    q = (q * sm_scale * 1.44269504).to(dtype)
    m_i = -inf, l_i = 0, acc = 0

    # 2) 段一：扫 slash 拼出的 K block
    for block_index in range(num_blks):
        start_n = tl.load(blks_ptr + block_index)
        k = tl.load(K[..., start_n + offs_n])
        v = tl.load(V[..., start_n + offs_n])
        qk = tl.dot(q, k)
        # 因果 mask: cols <= offs_m
        ... online softmax update m_i, l_i, acc ...

    # 3) 段二：扫 vertical 零散列（每次 BLOCK_N 个 gather）
    for start_n in range(0, num_cols, BLOCK_N):
        cols = tl.load(column_index[start_n: start_n+BLOCK_N])
        k = tl.load(K[..., cols])     # gather
        v = tl.load(V[..., cols])     # gather
        qk = tl.dot(q, k)
        ... online softmax update ...

    # 4) 输出
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)
```

性能要点：
- **段一连续访问**：把若干相邻 slash 合并成一段连续 K，访存最优。
- **段二 gather 访问**：vertical 列散布全文，必须 gather，但数量远少于段一。
- **online softmax**：`m_i`, `l_i` 跨段联合维护，保证两段结合后等价于完整的 softmax。
- **`exp2 + log2(e)` 替换 `exp`**：避开 Triton 编译器对 `exp` CSE/LICM 失效问题。

#### 5.1.3 v3 版本 `pit_sparse_flash_attention_v3.py`

约 1500 行，包含 FA3 / Hopper 优化的多 stage pipeline、WGMMA、按 head 分发 kernel 配置等。生产部署优先用此版本（或 SGLang/vLLM 集成的版本）。

### 5.2 Block-Sparse kernel
文件：`minference/ops/block_sparse_flash_attention.py`

#### 5.2.1 索引构造 `_build_block_index`（L151）

```python
query_pool = query.reshape(B,H, -1, block_size_M, D).mean(-2)   # 64-block 均值
key_pool   = key  .reshape(B,H, -1, block_size_N, D).mean(-2)
p_pool     = einsum('bhmk,bhnk->bhmn', query_pool, key_pool)
# 因果 mask
p_pool     = p_pool.where(arange_M >= arange_N, -inf)
return torch.topk(p_pool, top_k, -1).indices.int().sort(-1).values
```

即把 Q/K 各自 64-block mean-pool 后做一次小规模 QK，每行取 top-k 个 K block。

#### 5.2.2 Triton kernel `_triton_block_sparse_attn_fwd_kernel`（L29）

逻辑比 vertical-slash 简单：只有一个 K-block 循环，按 `block_index[start_m]` 的顺序读 K/V 块、做 FA online softmax。每个 query block 只算 `min((start_m+1)*BLOCK_M/BLOCK_N, MAX_BLOCKS_PER_ROW)` 个 K 块。

### 5.3 Streaming kernel（A-shape）
文件：`minference/ops/streaming_kernel.py`（839 行）

实现"sink + sliding window"的 attention：

- `_attn_fwd_inner`（L26）：FA2 风格 inner loop，外加 `SLIDING_WINDOW / COMPLEMENT_SLIDING_WINDOW` 两套区间逻辑：
  - `SLIDING_WINDOW=True, COMPLEMENT=False`：只算 `[start_m - sliding_window_size + offset, start_m + offset]` 的 K 块。
  - `COMPLEMENT=True`：算窗口之外的 K 块（用于复合方案）。
- 通过 `IS_EVEN_M / IS_EVEN_N` 编译期常量分发出 padding-free / padding 两个 kernel 变体。

对应的 sink 由 wrapper 在 query 前端单独 prepend `n_init` 个固定 attention（即"A 字形"中左侧那一竖）。

### 5.4 Fallback：`flash_attn_triton.py`

无 `flash-attn` 包时使用，提供 `_flash_attn_triton_decoding` 作为通用 `flash_attn_func`。

---

## 6. CUDA kernel：稀疏索引展开

文件：`csrc/vertical_slash_index.cu` + `csrc/kernels.cpp`，通过 `setup.py` 的 `CUDAExtension` 编译为 `minference.cuda` 子模块。

### 6.1 为什么需要这一步？

Triton kernel 的每个 thread block 处理一个 query block（行），需要 O(1) 拿到自己要算的 K block 列表。
但 Python 给的只是「vertical 列下标」和「slash 对角线下标」两个稀疏描述；
直接在 Triton 里展开既麻烦又低效，因此抽出来用 CUDA 写。

### 6.2 输入 / 输出

```cpp
// Python 侧调用：
block_count, block_offset, column_count, column_index =
    convert_vertical_slash_indexes(seqlens, v_idx, s_idx,
                                   context_size, 64, 64);

// 形状：
// seqlens       [B]                                int32
// v_idx         [B, H, NNZ_V]                      int32, 升序
// s_idx         [B, H, NNZ_S]                      int32, 降序
// ↓
// block_count   [B, H, num_M_rows]                 该 query block 需要算的 K-block 数
// block_offset  [B, H, num_M_rows, NNZ_S]          这些 K-block 的起始 N 偏移
// column_count  [B, H, num_M_rows]                 该 query block 需要算的零散列数
// column_index  [B, H, num_M_rows, NNZ_V]          这些零散列的 N 下标
```

### 6.3 Kernel 设计 `convert_vertical_slash_indexes_kernel`（L27）

```
grid = (N_HEADS, BATCH, ceil(num_M_rows / 64))
block = (64)            // 每个 thread 处理一个 query block (M-row)
```

每个线程的逻辑（双指针扫 vertical + slash）：

```cpp
int v = 0, s = 0;
int v_idx = vertical_indexes[v++];
int s_idx = slash_indexes[s++];

// 把 slash_idx 翻译成 K 维上的起始位置（带 BLOCK_M 对齐）
while (s_idx >= end_m) s_idx = slash_indexes[s++];
s_idx = max(end_m - s_idx, BLOCK_SIZE_M);

int range_start = s_idx - BLOCK_SIZE_M, range_end = s_idx;
while (1) {
    if (v_idx < range_end) {
        // 这个 vertical 列已经被 slash 覆盖
        if (v_idx < range_start) {
            // 落在已合并 range 之外 → 当作零散列写入 column_index
            column_index[tmp_col_cnt++] = v_idx;
        }
        v_idx = (v < NNZ_V) ? vertical_indexes[v++] : end_m + BLOCK_M;
    } else {
        // 取下一条 slash
        s_idx = (s < NNZ_S) ? max(end_m - slash_indexes[s++], BLOCK_M) : ...;
        if (s_idx > range_end + BLOCK_M) {
            // 与当前 range 不再相邻 → 输出当前 range 的 K 块
            save_blocks(block_offset, range_start, range_end, BLOCK_N, tmp_blk_cnt);
            range_start = s_idx - BLOCK_M;  range_end = s_idx;
        } else if (s_idx > range_end) {
            range_end += BLOCK_M;           // 与当前 range 相邻 → 合并
        }
    }
}

block_count[0]  = tmp_blk_cnt;
column_count[0] = tmp_col_cnt;
```

核心思想：
1. **slash 合并**：因 slash 索引降序，相邻 slash 在 K 维上很可能相邻或重叠，把它们合并成连续 range 后再切成 `BLOCK_SIZE_N` 大小的 K block，写入 `block_offset`。这是访存友好的关键。
2. **vertical 去重**：若 vertical 列已落在某个 slash range 内部，就不重复写入；否则当作真正零散列写到 `column_index`。
3. **写入 count**：最终 `block_count` 与 `column_count` 告知 Triton kernel 要扫多少。

### 6.4 不同 block size 的特化

`vertical_slash_index.cu` 提供了 4 套 wrapper（同一 kernel，不同 grid/thread 配置）：

- `convert_vertical_slash_indexes_64x64`（默认）
- `_padding_64x64`（带 pad 处理）
- `_mergehead_64x64`（同 batch 内不同 head 共享 slash 时合并）
- `_padding_mergehead_64x64`

### 6.5 pybind 入口 `csrc/kernels.cpp`

```cpp
std::vector<at::Tensor> convert_vertical_slash_indexes(
    torch::Tensor seqlens,
    torch::Tensor vertical_indexes,
    torch::Tensor slash_indexes,
    int context_size, int block_size_M, int block_size_N);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convert_vertical_slash_indexes",
          &convert_vertical_slash_indexes,
          "dynamic sparse index function");
}
```

`setup.py`（L99–L108）以 `name="minference.cuda"` 编译；Python 侧通过 `from ..cuda import convert_vertical_slash_indexes` 引用。

---

## 7. 配置体系

### 7.1 best_pattern JSON 格式

```json
[                              ← layer 列表
  {                            ← 第 0 层
    "0":  ["vertical_and_slash", 1000, 6096, 0.9623],   ← head_id: [ty, v_size, s_size, score]
    "1":  ["vertical_and_slash", 1000, 6096, 0.9543],
    ...
    "31": ["vertical_and_slash", 3500, 100,  0.7321]
  },
  { ... 第 1 层 ... },
  ...
]
```

- score 是离线搜索时该模式对真实 dense attention 的近似得分（取决于搜索版本，越大越好或差异越小越好）。
- 推理时根本不读 score，只用前三项。

### 7.2 model2path

`minference/configs/model2path.py` 维护 `MODEL2PATH` 字典，将 HF 模型名映射到 `*.json` 路径。
未在表内的模型可由用户通过 `is_search=True` 自行搜索生成（见 `experiments/` 下的指引）。

---

## 8. 推理调用与端到端数据流

### 8.1 HF 调用

```python
from transformers import pipeline
from minference import MInference

pipe = pipeline("text-generation", model=model_name, torch_dtype="auto", device_map="auto")
minference_patch = MInference("minference", model_name)
pipe.model = minference_patch(pipe.model)        # ← 完成 monkey-patch
pipe(prompt, max_length=10)
```

### 8.2 vLLM 调用

```python
from vllm import LLM, SamplingParams
from minference import MInference

llm = LLM(model_name, enforce_eager=True,
          max_model_len=128_000, enable_chunked_prefill=False)
minference_patch = MInference("vllm", model_name)
llm = minference_patch(llm)
llm.generate(prompts, sampling_params)
```

### 8.3 只用 kernel

```python
from minference import (
    vertical_slash_sparse_attention,   # Vertical-Slash
    block_sparse_attention,            # Block-Sparse
    streaming_forward,                 # A-shape (sink + sliding window)
)

attn_output = vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)
attn_output = block_sparse_attention(q, k, v, topk)
attn_output = streaming_forward(q, k, v, init_num, local_window_num)
```

### 8.4 一次 prefill 的完整调用栈

```
LlamaForCausalLM.forward
└─ LlamaModel.forward (被 patch)
   └─ DecoderLayer.forward × num_layers
      └─ LlamaAttention.forward = minference_forward()
         ├─ init_minference_parameters()        # 加载 self.best_pattern
         ├─ QKV proj + RoPE + KV cache update + repeat_kv
         └─ for head in range(num_heads):       # 逐 head
              └─ gather_last_q_vertical_slash_topk_v4(q,k,v,head)
                 └─ vertical_and_slash_kernel(q,k,v, v_size, s_size)
                    ├─ qk = q[:,:,-64:,:] @ k.T / sqrt(d)
                    ├─ softmax → 列和 / 反对角线和 → topk
                    └─ vertical_slash_sparse_attention(q,k,v, v_idx, s_idx)
                       ├─ pad + 排序索引
                       ├─ minference.cuda.convert_vertical_slash_indexes
                       │     ↳ CUDA kernel: 合并 slash range / 去重 vertical 列
                       └─ sgl_kernel/vllm sparse_attn_func          ← 优先
                         （或 fallback 到 _triton_mixed_sparse_attn_fwd_kernel）
                            ↳ Triton kernel:
                                段 1：扫 slash 拼出的 K 块（连续访问）
                                段 2：gather vertical 零散列
                                online softmax → acc / l_i → 输出
```

decoding（`q_len == 1`）短路成 dense `flash_attn_func`。

---

## 9. 构建与依赖

- `setup.py` 用 `torch.utils.cpp_extension.CUDAExtension` 编译：
  ```python
  CUDAExtension(
      name="minference.cuda",
      sources=["csrc/kernels.cpp", "csrc/vertical_slash_index.cu"],
      extra_compile_args=["-std=c++17", "-O3"],
  )
  ```
- 运行依赖：`transformers>=4.37.0`、`torch`、`triton`、`einops`；可选 `flash-attn` / `sgl_kernel` / `vllm_flash_attn`（无则 fallback 到 Triton）。
- 离线 wheel 与 fallback 源码构建逻辑参考 `setup.py` 的 `get_wheel_url` / `CachedWheelsCommand`。

---

## 10. 与其他模块的关系（仅速记）

- **kv_type**（独立维度）：`streamingllm` / `snapkv` / `pyramidkv` / `quest` / `kivi` / `retr_attn` / `leank`，可与 MInference 1.0 的 `attn_type` 任意组合，对应 SCBench 论文中"KV 生命周期"的另外几个阶段。实现位于 `minference/modules/kvcompression.py`、`snapkv.py`、`quest.py`、`pyramidkv.py`、`kivi.py`、`retr_attn.py`。
- **分布式**：`minference/dist_ops/minfer_striped.py`、`minfer_zigzag.py`、`minfer_dr_striped.py` 把 vertical-slash kernel 集成进 Ring/Striped/Zigzag 多卡 attention。
- **后继工作**：TriangleMix（`tri_mix` / `tri_shape`）、xAttention、FlexPrefill、LeanK 都复用了同一套 patch 框架与 op 接口，只是在 modules/ 下增加新的 forward 实现。

---

## 11. 调优 / 排错速查

| 现象 | 原因 / 解决 |
|---|---|
| 用了 minference 但速度没变 | 检查 `q_len`：MInference 1.0 只加速 prefill；q_len==1 直接 dense |
| 精度下降明显 | 调大 `vertical_size` / `slash_size`；或把 `starting_layer` 调高让前几层走 dense |
| `convert_vertical_slash_indexes` 报错 | 通常是 v_idx/s_idx 没排好序或 dtype 不是 int32；wrapper 已处理，自定义调用注意 |
| CUDA OOM at `_prepare_4d_causal_attention_mask_with_cache_position` | 给模型加 `_attn_implementation="flash_attention_2"` |
| CUDA OOM at lm_head | 设 `logits_to_keep=1` |
| 推理时不肯走 sparse | 检查 `model_name` 是否在 `MODEL2PATH`；自定义模型需 `is_search=True` 先生成 best_pattern |

---

## 12. 关键文件索引

| 角色 | 路径 |
|---|---|
| 顶层入口 | `minference/models_patch.py:19`（`MInference` 类） |
| 配置校验 | `minference/minference_configuration.py:7`（`MInferenceConfig`） |
| HF / vLLM patch | `minference/patch.py:1313`（`patch_hf`）、L1170+（`minference_patch_vllm*`） |
| Per-head 调度 | `minference/modules/minference_forward.py:296`（`gather_last_q_vertical_slash_topk_v4`） |
| 在线估计 | `minference/modules/minference_forward.py:381`（`vertical_and_slash_kernel`） |
| 反对角线求和 | `minference/modules/minference_forward.py:110`（`sum_all_diagonal_matrix`） |
| 离线搜索 | `minference/modules/minference_forward.py:129`（`search_pattern`）、L229（`_v2`） |
| Vertical-Slash op | `minference/ops/pit_sparse_flash_attention_v2.py:195` |
| Vertical-Slash kernel | `minference/ops/pit_sparse_flash_attention_v2.py:48`（`_triton_mixed_sparse_attn_fwd_kernel`） |
| Block-Sparse op | `minference/ops/block_sparse_flash_attention.py:169` |
| Block-Sparse kernel | `minference/ops/block_sparse_flash_attention.py:29` |
| Streaming kernel | `minference/ops/streaming_kernel.py:26`（`_attn_fwd_inner`） |
| CUDA 索引展开 | `csrc/vertical_slash_index.cu:27`（kernel）、L127（C++ wrapper） |
| pybind 导出 | `csrc/kernels.cpp:7` |
| Best-pattern 示例 | `minference/configs/Llama_3.1_8B_Instruct_128k_kv_out_v32_fit_o_best_pattern.json` |
