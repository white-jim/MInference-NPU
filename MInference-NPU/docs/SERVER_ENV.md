# 服务器运行备忘

## 固定环境

```bash
cd /data/guoshiyao/zhw/MInference-NPU/MInference-NPU
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python <cmd>
```

要点：

- CANN: `~/ascend/cann/8.5.0/cann-8.5.0`
- conda env: `flexhead-tl`
- TileLang source: `~/tilelang-ascend`
- local Phi3: `/data/guoshiyao/resources/models/Phi-3-mini-128k-instruct`
- HF cache: `/data/guoshiyao/resources/.hf_cache`

## 常用命令

生成 Phi3 probe configs：

```bash
python benchmarks/prepare_phi3_pathb_configs.py
```

HF clean stream profile：

```bash
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python examples/run_hf_minimal.py \
  --config-path minference/configs/Phi_3_mini_128k_instruct_pathb_stream_llm_aligned_dense_others.json \
  --ctx-len 4096 \
  --max-new-tokens 1 \
  --attn-type minference \
  --profile-branches \
  --num-runs 2
```

Dense baseline：

```bash
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python examples/run_hf_minimal.py \
  --ctx-len 4096 \
  --max-new-tokens 1 \
  --attn-type dense
```

核心回归：

```bash
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python tests/test_tilelang_sparse_attention.py --case all
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python -m pytest tests/test_block_sparse_kernel.py -q
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python -m pytest tests/test_streaming_kernel.py -q
```

## 已知结论

- 当前 path-B 可以命中真实 forward。
- Dense baseline 仍远快于 path-B probe。
- Clean probe 已排除 VS 干扰，瓶颈在 grouped TileLang path-B wrapper/kernel。
- `--profile-branches` 会同步 NPU，只用于定位，不作为普通性能数。
- `--num-runs 2` 用于区分 first-call/JIT 和 steady-state。

## 显存口径

`bench_tilelang_long_context.py` 里的 MB 是 isolated synthetic kernel allocator peak，只代表单 kernel benchmark 的局部峰值，不是模型总显存。不要用它解释长文本端到端显存。
