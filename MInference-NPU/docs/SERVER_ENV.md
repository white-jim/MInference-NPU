# 服务器环境与运行备忘

> 更新：2026-05-26。本文记录当前 NPU 服务器上继续 PR-4 TileLang 工作时必须固定的环境与命令。

## 1. 工作目录

源码仓库：

```bash
cd /data/guoshiyao/zhw/MInference-NPU/MInference-NPU
```

外层检查点文档在：

```bash
/data/guoshiyao/zhw/MInference-NPU/docs/context_checkpoint.md
```

## 2. 必需环境

当前 PR-4 TileLang 路径使用：

- CANN：`~/ascend/cann/8.5.0/cann-8.5.0`
- conda env：`flexhead-tl`
- TileLang 源码：`~/tilelang-ascend`
- 运行前必须 `source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh`
- 运行 TileLang 测试必须带 `PYTHONPATH=~/tilelang-ascend`

推荐统一命令模板：

```bash
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=~/tilelang-ascend conda run -n flexhead-tl python <script-or-command>
```

如果开新 bash 会话，仍需重新 source。为了减少漏配，可以在当前 shell 中定义：

```bash
run_flexhead_tl() {
  source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
  PYTHONPATH=~/tilelang-ascend conda run -n flexhead-tl "$@"
}
```

之后可运行：

```bash
run_flexhead_tl python tests/test_tilelang_sparse_attention.py --case all
```

## 3. 常用验证命令

TileLang separate-Q/K/V sparse attention：

```bash
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=~/tilelang-ascend conda run -n flexhead-tl python tests/test_tilelang_sparse_attention.py --case all
```

TileLang indices CPU 单测：

```bash
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=~/tilelang-ascend conda run -n flexhead-tl python -m pytest tests/test_tilelang_indices.py -v
```

Block-sparse 回归：

```bash
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=~/tilelang-ascend conda run -n flexhead-tl python -m pytest tests/test_block_sparse_kernel.py -q
```

TileLang 长上下文速度 / 显存 benchmark（先短后长）：

```bash
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=~/tilelang-ascend conda run -n flexhead-tl python benchmarks/bench_tilelang_long_context.py \
  --seq-lens 4096 8192 16384 \
  --iters 3 --warmup 1 \
  --modes block_sparse stream_llm \
  --json-output benchmarks/results/tilelang_long_context_short.json
```

官方 packed-KV SFA 闸门：

```bash
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=~/tilelang-ascend conda run -n flexhead-tl python tests/test_tilelang_sfa_integration.py --case all
```

## 4. 当前已验证结果

- `tests/test_tilelang_sparse_attention.py --case all`：6 个 case PASS，包括 `h1-one-block` 与 `h1-stream-llm-pad`。
- `block_sparse_attention` production `H==1` TileLang 入口 smoke：`max_abs_diff=9.7656e-04`, `mean_abs_diff=2.5847e-05`。
- `streaming_forward` production `H==1` TileLang 入口 smoke：`max_abs_diff=9.7656e-04`, `mean_abs_diff=2.9009e-05`。
- `python -m pytest tests/test_block_sparse_kernel.py -q`：29 passed。
- `python -m pytest tests/test_streaming_kernel.py -q`：32 passed。
- `benchmarks/bench_tilelang_long_context.py --seq-lens 4096 8192 16384`：path-B 全命中；16K steady-state：block_sparse 290.57 ms / synthetic kernel peak 232.0 MB，stream_llm 315.72 ms / synthetic kernel peak 74.6 MB。
- `benchmarks/bench_tilelang_long_context.py --seq-lens 32768 65536`：path-B 全命中；64K steady-state：block_sparse 1160.34 ms / synthetic kernel peak 928.1 MB，stream_llm 1226.30 ms / synthetic kernel peak 296.6 MB。

## 5. 注意事项

- `flexhead-tl` 环境可能没有完整 transformers 依赖；需要避免通过 `import minference` 触发 `minference/__init__.py` 的 eager import。相关测试已改成按文件路径 standalone load。
- TileLang kernel builder 里调用了 `tilelang.disable_cache()`，用于避免失败 JIT 版本污染后续动态 shape 绑定。
- 当前 PR-4 MVP 只接 `block_sparse` 与后续 `stream_llm` / A-shape；`vertical_and_slash` 暂不进入本阶段。
- 当前 separate-Q/K/V TileLang kernel 仍限制 `kv_group == 1`、causal fp16、`topk % block_I == 0`。
- 这个服务器上 `torch.npu.is_available()` 在部分会话会因为 `ascend_hal device count` 返回 False；已有测试和 benchmark 改用实际分配 `torch.empty(..., device="npu:0")` 作为可用性探针。
- benchmark 里 first-call 通常包含 TileLang 编译，可能是 14-16s；速度判断看 warmup 后的 `mean_ms`。
- `bench_tilelang_long_context.py` 的显存是 isolated synthetic kernel allocator 峰值，只能作为该 kernel benchmark 的局部指标；不包含模型权重、全层 KV cache、HF/transformers 常驻张量或 runtime reserve，不能解释为长文本模型推理总显存。
