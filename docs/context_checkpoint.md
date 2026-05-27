# Context Checkpoint

Last update: 2026-05-27

## Current Direction

Focus only on PR-4 TileLang path-B for Ascend NPU:

- `stream_llm`
- `block_sparse`
- Phi-3-mini-128k-instruct

Do not return to the removed old plans. The current tree is intentionally trimmed.

Backup before cleanup:

```text
/data/guoshiyao/zhw/MInference-NPU_backup_20260527_122449.tar.gz
```

## Runtime

```bash
cd /data/guoshiyao/zhw/MInference-NPU/MInference-NPU
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python <cmd>
```

Local model:

```text
/data/guoshiyao/resources/models/Phi-3-mini-128k-instruct
```

## Important Files

- `minference/modules/minference_forward.py`
  - best-pattern grouped scheduling
  - per-head `dense` pattern for clean probes

- `minference/ops/tilelang_sparse_attention.py`
  - separate Q/K/V TileLang sparse attention

- `minference/ops/streaming_kernel_npu.py`
  - `stream_llm` path-B wrapper

- `minference/ops/block_sparse_kernel_npu.py`
  - `block_sparse` path-B wrapper

- `benchmarks/prepare_phi3_pathb_configs.py`
  - generates the two compact Phi3 dense-others probe configs

- `examples/run_hf_minimal.py`
  - HF smoke runner
  - supports `--profile-branches`
  - supports `--num-runs`

## Current Results

Full probe vs dense baseline:

| ctx | stream probe | block probe | dense baseline |
|---:|---:|---:|---:|
| 4K | 22.46s | 22.26s | 0.60s |
| 8K | 30.48s | 30.18s | 0.97s |
| 16K | 49.98s | 50.98s | 2.32s |

Clean 4K probe, same process, `--num-runs 2`:

| config | run1 | run2 | branch timing |
|---|---:|---:|---|
| stream dense-others | 20.48s | 4.50s | `stream_llm: 24.042s / 12 calls`, `dense: 0.235s / 64 calls` |
| block dense-others | 20.03s | 3.50s | `block_sparse: 22.561s / 12 calls`, `dense: 0.249s / 64 calls` |

Interpretation:

- Path-B hits real HF forward.
- 43 target heads are grouped into 6 TileLang calls.
- First-call/JIT is large.
- Steady-state is still much slower than dense.
- Current bottleneck is grouped TileLang path-B wrapper/kernel.

## Next Step

Optimize the grouped TileLang path-B implementation:

1. Evaluate true multi-head / `kv_group>1` kernel support.
2. Evaluate a more efficient shared-index form.
3. Keep warmup/precompile as a separate first-call mitigation.
4. Do not expand to 32K/64K/128K until 4K/8K relative speed is meaningful.
