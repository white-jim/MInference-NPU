# PR-4 TileLang SFA 适配设计记录

> 更新：2026-05-26。范围只覆盖 `stream_llm` / A-shape 与 `block_sparse`；`vertical_and_slash` 暂不进入 PR-4 MVP。

## 1. 当前可用的官方 SFA 能力

`tilelang-ascend` 官方 `sparse_attention_fwd` 已在 `flexhead-tl` 环境跑通：

```python
sparse_attention_fwd(
    heads, dim, tail_dim, topk, kv_stride,
    kv_group=1, sm_scale=None, is_causal=True, block_I=64,
)
kernel(q, kv, indices)
```

- `q`: `[B, S_q, H, dim + tail_dim]`
- `kv`: `[B, S_k, kv_group, dim + tail_dim]`
- `indices`: `[B, S_q, kv_group, topk]`，元素是 K token 位置，不是 block id
- `out`: `[B, S_q, H, dim]`
- 当前 example 强约束 `kv_group == 1`
- 当前 kernel 对 pad sentinel 不容错，`-1` / `S_k` 都不能直接喂生产 kernel

三个闸门 case 已通过：sanity / block-sparse / stream-llm。

## 2. Pad 策略

短期策略：`tilelang_indices.py::sanitize_indices_for_tilelang_kernel`。

真实 A-shape 早期 token 可能产生无效槽位；官方 kernel 又不支持 pad。适配层把 pad 替换为同一 Q 行的第一个“未来 K token”：

- 地址合法，避免越界载入或 NaN。
- `is_causal=True` 会屏蔽该未来 token，因此不改变可见集合。
- 如果某行已经没有未来 token 但仍存在 pad，函数直接抛 `ValueError`。这代表 fixed-topk 官方 kernel 无法无损表达该行，需要改 kernel 原生支持 pad，或调整 indices 构造保证该行无 pad。

CPU 单测已验证 sanitize 前后的 causal-visible set 完全一致。

## 3. Q 窗口绝对位置

`stream_llm_to_tilelang` 新增 `q_start_index_s`。

原因：官方 SFA reference 默认 `q_start_index_s = S_k * kv_stride - S_q`，即 Q 是 KV 尾部窗口。旧实现默认按 full prefill 的 `q_start=0` 构造 local window，在尾部窗口探针里语义不严谨。

现在：

- full prefill：`q_start_index_s=0`
- 官方尾部窗口：`q_start_index_s=S_k * kv_stride - S_q`
- local 段按绝对位置 `[q_abs - n_local + 1, q_abs]` 构造

## 4. `kv_group == 1` 策略

PR-4 MVP 采用共享 pattern：

- 官方 example 当前只支持 `kv_group=1`，即所有 Q heads 共享同一组 KV / indices。
- `block_sparse` 与 `stream_llm` 可以先用共享 indices 跑通真稀疏链路。
- per-head pattern、MHA 下每 head 独立 indices、GQA 下 per-group union 都暂不进入 MVP。

后续若继续复用官方 kernel，需要二选一：

1. 修改 tilelang kernel 支持 `kv_group > 1`。
2. 在 Python 侧做 per-group union / shared pattern，把多个 head 合并到一组 indices。

## 5. 标准 K/V 与 packed KV

官方 SFA 不是 MInference 标准 `q, k, v` 的 drop-in 替代。

官方语义：

- `k = kv`
- `v = kv[..., :dim]`
- `q/k` last dim 是 `dim + tail_dim`
- 输出 V 维度是 `dim`

MInference 标准语义：

- `q, k, v` 分离
- `q/k/v` head_dim 通常相同
- `v` 不是 `k` 的前缀切片

因此官方 SFA 只能作为接口 / indices / tilelang 工具链闸门。要保持普通 attention 精确语义，PR-4-BS / PR-4-SL 的生产实现应写一个更贴近 MInference 的 tilelang kernel，直接接受分离的 `q, k, v, indices`。

## 6. 分离 Q/K/V MVP 状态

已新增 `minference/ops/tilelang_sparse_attention.py`：

- `build_sparse_attention_qkv_fwd(...)`：构建 `kernel(q, k, v, indices)`，输入为 BSHD/BSGD，输出 BSHD。
- `sparse_attention_qkv_reference(...)`：同接口的 PyTorch reference。

当前 MVP 约束：

- `q`: `[B, S_q, H, D]`
- `k/v`: `[B, S_k, kv_group, D]`
- `indices`: `[B, S_q, kv_group, topk]`
- `kv_group == 1`
- causal fp16 forward only
- `topk % block_I == 0`
- indices 槽位可使用 `-1` 作为 pad sentinel；kernel 会在载入 K/V 前跳过 pad，并把对应 logits mask 为 `-inf`

验证脚本：`tests/test_tilelang_sparse_attention.py`。

实机结果（`flexhead-tl` + `PYTHONPATH=~/tilelang-ascend`）：

| case | 语义 | 结果 |
|---|---|---|
| one-block | 每行只看 K block 0 | PASS，`max_abs_diff=9.7656e-04` |
| two-block | indices 包含未来 block，kernel causal mask 裁未来 token | PASS，`max_abs_diff=9.7656e-04` |
| h1-one-block | production per-head 形态，`H==1` 小 head padding | PASS，`max_abs_diff=9.7656e-04` |
| stream-llm | `stream_llm_to_tilelang(q_start_index_s=384)` 尾部窗口，无 pad | PASS，`max_abs_diff=4.8828e-04` |
| stream-llm-pad | full-prefill 早期 token，`-1` pad 槽原生跳过 | PASS，`pad_count=6112`, `max_abs_diff=9.7656e-04` |
| h1-stream-llm-pad | production per-head A-shape，`H==1` + `-1` pad | PASS，`max_abs_diff=9.7656e-04` |

两个实现细节：

- TileLang 的 `T.prim_func` 不能放在启用了 `from __future__ import annotations` 的文件里，否则 `T.Tensor(...)` 注解会被延迟成字符串并触发 TVMScript parser 错误。
- 本 kernel builder 调用 `tilelang.disable_cache()`。调试过程中同名 JIT factory 的失败版本会污染动态 shape 绑定，表现为 `KeyError: 'Unfounded symbolic var: batch'`。
- `build_sparse_attention_qkv_fwd` 现在有进程内 callable cache，key 为 `(heads, dim, topk, kv_group, sm_scale, causal, block_I, q_start, dtype, core_num)`。否则 benchmark 会反复把 TileLang JIT 编译时间算进每次 forward。
- `H==1` 已按官方小 H padding 思路修复：`head_kv` pad 到至少 16，`kv_group==1` 下多出来的 head 由 TileLang 边界处理忽略，避免 `H_per_block // 2 == 0`。
- `block_sparse_attention` 的生产 per-head 入口已接入 TileLang 路径：`H==1 + fp16 + NPU` 时从 `_select_block_sparse_topk_indices` 构造 TileLang token indices，超出 `S_k` 的 padding token 改成 `-1`，再调用 separate-Q/K/V kernel。实机 smoke：`tilelang_entry max_abs_diff=9.7656e-04, mean_abs_diff=2.5847e-05`。
- `streaming_forward` 的生产 per-head 入口已接入 TileLang 路径：`H==1 + fp16 + NPU` 且 `(n_init,n_local)` 64 对齐时用 `stream_llm_to_tilelang(q_start_index_s=S_k-S_q)` 生成 A-shape indices。实机 smoke：`tilelang_stream_entry max_abs_diff=9.7656e-04, mean_abs_diff=2.9009e-05`。
- 保留旧 path-A 作为非 MVP 兜底：`H!=1`、非 fp16、非 NPU、TileLang 未安装、或 A-shape 参数未 64 对齐时仍走原 `npu_fusion_attention + bool mask` / PyTorch reference；TileLang 编译/运行错误不吞掉，继续显式暴露。

## 7. 速度 / 显存 benchmark

新增脚本：`benchmarks/bench_tilelang_long_context.py`。

用途：

- 直接 standalone 加载 op 文件，避免 `flexhead-tl` 缺 transformers 时触发顶层 `minference` import。
- 记录 first-call/JIT 时间、steady-state 平均耗时、tokens/s、path-B 命中次数、NPU peak allocated/reserved。
- 当前 benchmark 只测 production per-head 形态：`B=1, H=1, D=64, fp16, topk_tokens=1024`。
- **重要限制**：这里的显存是 synthetic kernel micro-benchmark 的 allocator 峰值，只包含单个 `H==1` q/k/v 输入和该 op 的临时张量 / workspace；不包含模型权重、全层 KV cache、完整多 head 调度、runtime 常驻 reserve、allocator 碎片或 HF/transformers 端到端开销。因此表里的 MB 只能作为该 kernel benchmark 的局部指标，不能解释为“长文本模型推理总显存”。

实测环境：NPU 服务器 `flexhead-tl`，CANN 8.5.0，`PYTHONPATH=~/tilelang-ascend`。命令：

```bash
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=~/tilelang-ascend conda run -n flexhead-tl python benchmarks/bench_tilelang_long_context.py \
  --seq-lens 4096 8192 16384 \
  --iters 3 --warmup 1 \
  --modes block_sparse stream_llm \
  --json-output benchmarks/results/tilelang_long_context_short.json
```

短 / 中长度结果：

| mode | S | topk tokens | first call ms | mean ms | tokens/s | path-B hits | synthetic peak alloc MB | synthetic peak reserved MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| block_sparse | 4K | 1024 | 15956.03 | 73.36 | 55832.2 | 1/3 | 58.0 | 90.0 |
| stream_llm | 4K | 1024 | 14610.46 | 93.64 | 43740.4 | 1/3 | 19.1 | 22.0 |
| block_sparse | 8K | 1024 | 148.09 | 145.06 | 56474.2 | 1/3 | 118.0 | 140.0 |
| stream_llm | 8K | 1024 | 205.45 | 163.42 | 50127.4 | 1/3 | 38.6 | 56.0 |
| block_sparse | 16K | 1024 | 297.02 | 290.57 | 56385.0 | 1/3 | 232.0 | 366.0 |
| stream_llm | 16K | 1024 | 379.14 | 315.72 | 51893.3 | 1/3 | 74.6 | 88.0 |
| block_sparse | 32K | 1024 | 16475.43 | 577.69 | 56722.6 | 1/3 | 464.1 | 686.0 |
| stream_llm | 32K | 1024 | 15201.17 | 611.84 | 53556.6 | 1/3 | 148.6 | 172.0 |
| block_sparse | 64K | 1024 | 1158.71 | 1160.34 | 56480.0 | 1/3 | 928.1 | 1344.0 |
| stream_llm | 64K | 1024 | 1304.80 | 1226.30 | 53442.2 | 1/3 | 296.6 | 318.0 |

观察：

- 4K 的 first-call 包含 TileLang 编译，约 14-16s；steady-state 需看 `mean_ms`。
- 32K 那轮在新进程重新编译，因此 first-call 再次约 15-16s；64K 已复用同进程 cache。
- 4K 到 64K steady-state 基本线性，tokens/s 稳在约 53K-56K。
- synthetic kernel 显存随 S 近似线性增长；这只能说明 path-B op 层面符合 `O(S * topk)` 预期，不再是 path-A bool mask 的 `O(S^2)`。
- 当前还只是 `H==1` per-head 生产形态。完整模型若仍逐 head 调用，会把这个耗时乘上 head/layer 调度次数；模型运行时还会有权重、KV cache、框架常驻内存等额外占用。下一步重点仍是 batched/shared-index 调度或 HF 小规模 smoke 来定位端到端瓶颈。

## 8. 下一步

1. 先补 8K/16K/32K 的端到端 HF smoke，确认真实 `best_pattern` forward 会命中 TileLang path-B，并量化 per-head 调度开销。
2. 评估是否把 block-sparse / stream-llm 从 per-head 调度升级成 batched/shared-index 调度，减少 kernel launch 次数。
3. 继续按台阶测更长上下文：短长度稳定后先 128K，再 256K；记录速度、显存和是否 fallback。
4. 再评估是否把 `kv_group==1` 扩展到 per-head / per-group K/V。
