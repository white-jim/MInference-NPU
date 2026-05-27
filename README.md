# MInference-NPU

当前工作区只保留 PR-4 方向：在 Ascend NPU 上验证并优化 Phi-3 的 TileLang path-B 稀疏注意力。

## 当前目标

- 只聚焦 `stream_llm` 和 `block_sparse`。
- 用 Phi-3-mini-128k-instruct 做 4K/8K/16K 短阶梯验证。
- 速度必须始终和 `--attn-type dense` baseline 对比。
- 暂不继续扩到 128K/256K，直到短长度下的端到端瓶颈清楚。

## 目录

```text
MInference-NPU/                 # 代码仓库
  minference/                   # NPU patch 与算子
  benchmarks/                   # Phi3 probe config 生成、kernel benchmark
  examples/run_hf_minimal.py    # HF smoke / 分支计时入口
  docs/                         # 只保留当前路线文档
docs/context_checkpoint.md      # 短上下文检查点
```

完整上游副本、早期迭代文档和历史日志已删除。删除前备份位于：

```text
/data/guoshiyao/zhw/MInference-NPU_backup_20260527_122449.tar.gz
```

## 快速命令

```bash
cd /data/guoshiyao/zhw/MInference-NPU/MInference-NPU
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python examples/run_hf_minimal.py \
  --config-path minference/configs/Phi_3_mini_128k_instruct_pathb_stream_llm_aligned_dense_others.json \
  --ctx-len 4096 \
  --max-new-tokens 1 \
  --attn-type minference \
  --profile-branches \
  --num-runs 2
```

## 最新结论

- Phi3 probe path-B 可以命中，43 个目标 heads 会聚合为 6 次 grouped TileLang 调用。
- Dense baseline 明显更快：4K dense 约 `0.60s`。
- Clean probe steady-state 仍慢：stream 4K 第二轮约 `4.50s`，block 约 `3.50s`。
- 下一步应优化 grouped TileLang wrapper/kernel，而不是继续加长上下文测试。
