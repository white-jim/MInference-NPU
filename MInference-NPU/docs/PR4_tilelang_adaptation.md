# PR-4 当前状态（2026-05-27 晚）

## 当前目标

两条真稀疏路径，状态已经明确分叉：

- `stream_llm` —— **已完成从 TileLang 到 Ascend hardware band + sink + LSE merge 的迁移**。
- `block_sparse` —— 仍是 TileLang path-B，下一阶段重点改造。

验证模型固定 Phi-3-mini-128k-instruct。当前 stream probe 只覆盖 43 / 1024 heads，因此它能证明 `stream_llm` kernel 迁移成功，但不能单独带来显著端到端加速。下一阶段要扩大稀疏覆盖面，重点在 `block_sparse`。

## stream_llm 阶段结论

### 改造前问题

4K clean probe 中，stream 走 TileLang path-B 的 steady-state run2 约 `4.50s`，dense baseline 约 `0.60s`。逐层拆解约为 6 个 active layer x `0.67s/layer`，比 dense 同层 attention 慢约 200x。

根因是结构性错配：

1. TileLang kernel 中 `padded_head_kv = max(next_pow2(h), 16)`，调用方 fold 后 `heads=1, kv_group=1`，导致每个 core 处理 16 个 head 槽，15 个是无效计算。
2. 通用 sparse kernel 对 K/V 做 per-token gather；stream_llm 的访问本来是 sink + sliding 两段连续区间，应使用 range/band 访问。
3. wrapper 强制 fold `B*H` 到 batch 维，无法表达真实 MHA/GQA 的 H 维结构。

### 新实现

`stream_llm` 完全退出 TileLang：

1. Sliding window：`torch_npu.npu_fusion_attention(sparse_mode=4, pre_tockens=n_local-1, next_tockens=0, atten_mask=compressed_2048x2048_causal_mask)`。
2. Sink：`sparse_mode=1` 跑 `K_len=n_init` 的小 dense FA + bool mask。
3. LSE merge：用两段返回的 `softmax_max` / `softmax_sum` 合并。

关键坑：

- `sparse_mode=4` 必须传入 2048x2048 compressed causal mask；`atten_mask=None` 时 torch_npu 会忽略 sparse 参数并退化成 full attention。
- `S_q < S_k` 时，band FA 不自动按绝对 KV 位置对齐；当前实现对 Q 补 dummy prefix 后切回尾部结果。

### 正确性

- `tests/test_streaming_kernel.py -q`：32 passed。
- `tests/test_block_sparse_kernel.py -q`：29 passed。
- 64K isolated stream kernel：`S=65536, H=4, D=128, n_init=128, n_local=3968`，抽样行与 CPU fp32 streaming-LLM 定义对比，最大误差 `9.77e-4`。

注意：这个误差是对 streaming-LLM 稀疏定义本身的 fp16 kernel 数值误差，不是相对 full dense attention 的模型质量损失。

### 性能

当前 Phi-3 stream dense-others probe 每 run 只有 43 heads 走 `stream_llm`，981 heads 仍为 dense-others。branch timing 是 attention 分支累计时间。

| ctx | stream probe run2 | dense branch | stream_llm branch |
|---:|---:|---:|---:|
| 4K | 0.31s | 0.235s / 64 calls | 0.048s / 12 calls |
| 8K | 0.70s | 0.654s / 64 calls | 0.049s / 12 calls |
| 16K | 2.03s | 2.376s / 64 calls | 0.057s / 12 calls |
| 32K | 6.52s | 9.477s / 64 calls | 0.076s / 12 calls |
| 64K | 28.88s（2 NPU, `device_map=auto`） | 18.066s / 64 calls | 0.122s / 12 calls |

64K 平均每 head attention 分支耗时：

- dense-others：`18.066s / (2 * 981 heads) = 9.21ms/head`
- stream_llm：`0.122s / (2 * 43 heads) = 1.42ms/head`
- 算子分支 per-head 加速约 `6.49x`

### 端到端对比

同口径 dense baseline：Phi-3-mini-128k-instruct，本地权重，`max_new_tokens=1`，`--num-runs 2`，比较 run2。64K 使用 2 NPU + `device_map=auto`，其余为单 NPU。

| ctx | dense run2 | stream probe run2 | E2E speedup |
|---:|---:|---:|---:|
| 4K | 0.28s | 0.31s | 0.90x |
| 8K | 0.69s | 0.70s | 0.99x |
| 16K | 2.05s | 2.03s | 1.01x |
| 32K | 6.67s | 6.52s | 1.02x |
| 64K | 29.61s | 28.88s | 1.03x |

结论：`stream_llm` 算子迁移成功，但当前 probe 没有实质端到端加速。原因是稀疏覆盖率太低，端到端主要被 dense-others、HF/Phi-3 4D causal mask 构造、QKV/O projection 和调度/同步开销主导。

## 下一阶段：block_sparse

目标：把 stream_llm 阶段学到的经验迁移到数据相关稀疏路径，扩大稀疏覆盖面，争取真实端到端收益。

`block_sparse` 与 `stream_llm` 的区别：

- indices 来自 pooled QK top-k，数据相关。
- hardware band 不适用。
- key 位置以 block 为粒度天然连续，适合改成 per-block range load，而不是 per-token gather。

优先方向：

1. 重写/改造 TileLang kernel 支持真实 H 维，不再把 `B*H` fold 到 batch。
2. 去掉 small-H padding 到 16 后的无效计算，或至少让一次 launch 处理真实多个 head。
3. K/V 内层从单 token gather 改成 per-block range load。
4. 保留短序列 `sparse_mode=1 + bool mask` 作为过渡对照，但不能依赖它解决 32K+。

## 常用验证命令

```bash
cd /data/guoshiyao/zhw/MInference-NPU/MInference-NPU
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh

PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python -m pytest tests/test_streaming_kernel.py -q
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python -m pytest tests/test_block_sparse_kernel.py -q

PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python examples/run_hf_minimal.py \
  --model microsoft/Phi-3-mini-128k-instruct \
  --model-path /data/guoshiyao/resources/models/Phi-3-mini-128k-instruct \
  --config-path minference/configs/Phi_3_mini_128k_instruct_pathb_stream_llm_aligned_dense_others.json \
  --ctx-len 32768 --max-new-tokens 1 --attn-type minference \
  --profile-branches --num-runs 2
```
