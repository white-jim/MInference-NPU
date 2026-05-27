# PR-4 当前状态（2026-05-27 晚）

## 当前目标

两条真稀疏路径，状态已分叉：

- `stream_llm` —— **2026-05-27 已弃用 TileLang，改 hardware band + sink + LSE merge**
- `block_sparse` —— 仍是 TileLang path-B（未动，等 stream_llm 跑通验证瓶颈判断后再改）

验证模型固定 Phi-3-mini-128k-instruct。短长度跑通并看清速度瓶颈前，不推进 128K/256K。

## 性能问题诊断（2026-05-27）

### 改造前实测

| ctx | stream probe | block probe | dense baseline |
|---:|---:|---:|---:|
| 4K | 22.46s | 22.26s | 0.60s |
| 8K | 30.48s | 30.18s | 0.97s |
| 16K | 49.98s | 50.98s | 2.32s |

Clean 4K，`--num-runs 2`：

| config | run1 | run2 | branch timing |
|---|---:|---:|---|
| stream dense-others | 20.48s | 4.50s | `stream_llm: 24.042s / 12 calls`, `dense: 0.235s / 64 calls` |
| block dense-others | 20.03s | 3.50s | `block_sparse: 22.561s / 12 calls`, `dense: 0.249s / 64 calls` |

去 JIT 后稳态仍 ~7.5× 慢于 dense。单层稀疏 attention ~0.67s vs dense 同层 ~0.003s，**慢 ~200×**。

### 根因（三层叠加）

1. **kernel 16× 算力浪费**
   `tilelang_sparse_attention.py:137`：`padded_head_kv = max(next_pow2(h), 16)`。wrapper 调用时 `heads=1, kv_group=1` → padded=16 → `H_per_block=16`。每个 core 处理 16 个 head 槽，只有第 0 个有真数据。L1/L0C 上 `acc_s_l0c[16,64]`、`acc_o_l0c[16,128]` 全被算了，结果只取 1 行。
2. **K/V 单 token gather**
   `tilelang_sparse_attention.py:270-281` 内层：
   ```python
   for bi_i in range(BI // 2):
       pos = indices_ub[bi_i + vid * BI // 2]
       T.copy(K[b_i, pos, g_i, :D], k_ub)   # 单 token HBM 散读
   ```
   每个 query 1024 次散读（topk=1024），256B/次，不吃 burst 带宽。
   stream_llm 的 K 位置本来就是两段**连续区间**（`[0:n_init]` + `[abs_i-n_local+1:abs_i+1]`），用通用 per-token gather kernel 是**抽象错配**。
3. **wrapper 强制 fold-into-batch**
   `streaming_kernel_npu.py` 和 `block_sparse_kernel_npu.py` 都把 `B*H` 折叠到 batch 维，因为 kernel 的 `kv_group=1` MVP + `padded_head_kv != head_kv and kv_group != 1` 断言导致真 MHA（H 个不同 K）无法直接表达。结果：问题 1 永远被放大 16×。

三层乘积 ≈ 实测的 200×。**不是参数能调好的，是结构性错配。**

## 已执行改动（stream_llm 重写）

### 决策

stream_llm 完全弃用 TileLang，改：

1. **Sliding window**：`torch_npu.npu_fusion_attention(sparse_mode=4, pre_tockens=n_local-1, next_tockens=0)` —— Ascend hardware band attention，**kernel 级真稀疏**，只算 band 内的 token。
2. **Sink**：`sparse_mode=1` 跑一个 K_len=n_init 的极小 dense FA + bool mask（mask 形状 `[S_q, n_init]`，n_init ≤ 256 不爆炸）。Mask 排除与 sliding window 的重叠位置。
3. **LSE merge**：两段返回的 `softmax_max` / `softmax_sum` 在线合并。

FLOPs ≈ `O(S × (n_init + n_local))`，与上游 streaming-LLM 一致。

### 与已弃用 path A 的关系

- path A = `sparse_mode=1 + 大 bool mask`：dense QK matmul + softmax 后 mask -inf，**算力没省**，O(S²) 显存，128K OOM。本次决策**不复辟**这条。
- 本路径 sliding window 走 `sparse_mode=4`（硬件 band，真稀疏）；sink 走 `sparse_mode=1` 但 K_len 极小（n_init ≤ 256），mask 不爆炸。
- 两者本质不同。

### 与 TileLang path-B 的关系

- TileLang sparse_attention_fwd = 通用 per-token gather kernel，对位置确定的 streaming 是 over-engineering（详见上面根因）。
- stream_llm 这条线**永久退出**TileLang 框架；block_sparse 仍留在 TileLang 里待重写。

### 代码改动

文件：

- `minference/ops/streaming_kernel_npu.py`
  - `_streaming_npu`：保留符号，实现换成 band + sink + LSE merge。
  - 删除 `_streaming_tilelang_npu`、`_load_sibling_module`、`_SIBLING_MODULE_CACHE`、`_TILELANG_BLOCK_SIZE`、相关 import。
  - 删除 `n_init % 64 == 0 / n_local % 64 == 0` 约束（band 可任意 int）。
  - `streaming_forward` 主路径不再 try/except TileLang，直接调 `_streaming_npu`；非 NPU 仍走 PyTorch ref。
- `examples/run_hf_minimal.py`：profile 钩子从 `_streaming_tilelang_npu` 改挂到 `_streaming_npu`。
- `block_sparse_kernel_npu.py` 与 `tilelang_*.py` 未动。

公开 API（`streaming_forward` 签名、`_streaming_npu` 符号、`_streaming_pytorch_ref` 黄金参考）全部保留，单测理论上不需要改。

## 待验证（服务器执行）

按以下顺序跑。完整 stdout 贴回，根据数字决定下一步。

### 1. 单元测试（数值正确性）

```bash
cd /data/guoshiyao/zhw/MInference-NPU/MInference-NPU
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python -m pytest tests/test_streaming_kernel.py -v
```

期望：所有 CPU+NPU 用例过。特别是 `test_npu_vs_pytorch_ref` 三个 case max_abs_diff < 1e-2。
失败 → band 语义（`pre_tockens` 拼写 / 行列对齐）或 LSE merge 有问题。

### 2. block_sparse 没破回归

```bash
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python -m pytest tests/test_block_sparse_kernel.py -q
```

### 3. 4K stream probe 性能（核心目标）

```bash
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python examples/run_hf_minimal.py \
  --model microsoft/Phi-3-mini-128k-instruct \
  --model-path /data/guoshiyao/resources/models/Phi-3-mini-128k-instruct \
  --config-path minference/configs/Phi_3_mini_128k_instruct_pathb_stream_llm_aligned_dense_others.json \
  --ctx-len 4096 --max-new-tokens 1 --attn-type minference \
  --profile-branches --num-runs 2
```

期望：
- run1 显著 < 之前 20.48s（无 TileLang JIT）
- run2 稳态 < 1s（vs 改造前 4.50s）
- branch timing：`stream_llm` < 0.5s / 12 calls

### 4. dense baseline 对照

```bash
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python examples/run_hf_minimal.py \
  --model-path /data/guoshiyao/resources/models/Phi-3-mini-128k-instruct \
  --ctx-len 4096 --max-new-tokens 1 --attn-type dense
```

应保持 ~0.60s。和测试 3 run2 对比 —— stream probe **应 ≤ dense**（FLOPs 比 dense 少 4×）。

### 5. 8K / 16K 趋势（可选）

4K 通过后再跑：换 `--ctx-len 8192` 和 `--ctx-len 16384`。期望随 S 增大 sparse 相对优势更大。

## 性能门槛

- 4K stream attention 一层 < 0.01s（vs 改造前 0.67s）
- 端到端 4K stream probe **稳态 ≤ dense baseline**

未达标禁止推进 32K/64K/128K。

## block_sparse 待办（下一阶段）

stream_llm 跑通且诊断验证正确后启动。两个候选方向：

| # | 方向 | 工作量 | 评注 |
|---|---|---|---|
| 1 | 重写 TileLang kernel 支持 MHA：去掉 `padded_head_kv != head_kv and kv_group != 1` 断言；K/V per-head 直传不 fold；内层把 token gather 改成 per-block range load（block_sparse 的 K 位置以 block 粒度天然连续） | 大 | 长期方向 |
| 2 | 短序列退 path A：4K-16K 内 `sparse_mode=1 + bool mask`，显存 256MiB 内可承受 | 小 | 过渡，≥32K 仍需 kernel |

方向 1 必须做。方向 2 可作 stream_llm 跑通后的快速兜底，不阻塞长期规划。具体策略等 stream_llm 验证结束后再立。
