# PR-4 TileLang Path-B 当前状态

## 目标

当前只做两条真稀疏路径：

- `stream_llm`
- `block_sparse`

验证模型固定为 Phi-3-mini-128k-instruct。短长度跑通并看清速度瓶颈前，不继续推进 128K/256K。

## 核心实现

- `minference/ops/tilelang_sparse_attention.py`
  - separate Q/K/V TileLang sparse attention
  - fp16 causal
  - native `-1` pad skip
  - kernel callable cache

- `minference/ops/streaming_kernel_npu.py`
  - `stream_llm` path-B
  - 同参数多 head 折叠到 batch 维做 grouped launch

- `minference/ops/block_sparse_kernel_npu.py`
  - `block_sparse` path-B
  - 同参数多 head 折叠到 batch 维做 grouped launch

- `minference/modules/minference_forward.py`
  - 按 pattern/参数分组调度
  - 支持 per-head `dense` pattern，用于 clean probe

## Phi3 Probe Config

由 `benchmarks/prepare_phi3_pathb_configs.py` 生成：

- `Phi_3_mini_128k_instruct_pathb_stream_llm_aligned_dense_others.json`
- `Phi_3_mini_128k_instruct_pathb_block_sparse_probe_dense_others.json`

这两个配置只让 43 个目标 heads 走 path-B，其余 heads 走 dense。

## 当前实测

命令模板：

```bash
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python examples/run_hf_minimal.py \
  --model microsoft/Phi-3-mini-128k-instruct \
  --model-path /data/guoshiyao/resources/models/Phi-3-mini-128k-instruct \
  --config-path minference/configs/Phi_3_mini_128k_instruct_pathb_stream_llm_aligned_dense_others.json \
  --ctx-len 4096 \
  --max-new-tokens 1 \
  --attn-type minference \
  --profile-branches \
  --num-runs 2
```

端到端 probe 与 dense baseline：

| ctx | stream probe | block probe | dense baseline |
|---:|---:|---:|---:|
| 4K | 22.46s | 22.26s | 0.60s |
| 8K | 30.48s | 30.18s | 0.97s |
| 16K | 49.98s | 50.98s | 2.32s |

Clean 4K profile：

| config | run1 | run2 | branch timing |
|---|---:|---:|---|
| stream dense-others | 20.48s | 4.50s | `stream_llm: 24.042s / 12 calls`, `dense: 0.235s / 64 calls` |
| block dense-others | 20.03s | 3.50s | `block_sparse: 22.561s / 12 calls`, `dense: 0.249s / 64 calls` |

## 结论

- path-B 已进入真实 HF forward。
- 43 个目标 heads 会聚合为 6 次 grouped TileLang 调用。
- 首次 TileLang JIT/first-call 是明显开销。
- 即使看第二轮 steady-state，stream/block 仍显著慢于 dense baseline。
- 当前瓶颈在 grouped TileLang wrapper/kernel 本身。

## 下一步

1. 评估真正 multi-head / `kv_group>1` TileLang kernel，避免把 H 折叠到 batch 后低效执行。
2. 评估 shared-index 形态，减少重复 indices / wrapper 调度开销。
3. 保留 warmup / 预编译策略，降低 first-call JIT 污染。
4. 只有短长度相对 dense 的关系合理后，再推进 32K/64K 和 128K/256K。
