# MInference-NPU

当前工作区只保留 PR-4 方向：在 Ascend NPU 上验证并优化 Phi-3 的真稀疏注意力。

## 当前目标

- 只聚焦 `stream_llm` 和 `block_sparse`。
- `stream_llm` 已从 TileLang 退出，改为 Ascend hardware band + sink + LSE merge。
- `block_sparse` 仍在 TileLang path-B，下一阶段重点改造。
- 用 Phi-3-mini-128k-instruct 做 4K/8K/16K/32K/64K 阶梯验证。
- 速度必须始终和 `--attn-type dense` baseline 对比。
- 当前 stream probe 只覆盖 43 / 1024 heads，不能代表全模型稀疏收益。

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

- `stream_llm` kernel 迁移成功：64K isolated stream kernel 抽样误差约 `9.77e-4`，64K probe 中 `stream_llm` branch 为 `0.122s / 12 calls`。
- 端到端暂未出现实质加速：64K dense run2 `29.61s`，stream probe run2 `28.88s`，约 `1.03x`。
- 原因是当前 probe 只有 43 个 heads 走 `stream_llm`，其余 981 个 heads 仍是 dense-others，端到端被 dense-others、HF/Phi-3 4D causal mask、QKV/O proj 等公共开销主导。
- 下一步应把 stream_llm 经验用于 `block_sparse`：扩大稀疏覆盖面，重写/优化 TileLang path-B，优先解决 fold-into-batch、16x small-H padding 浪费和单 token gather。
