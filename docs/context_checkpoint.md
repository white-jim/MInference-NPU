# Context Checkpoint

Last update: 2026-05-27（晚 — stream_llm 切 band+sink 后）

## Current Direction

PR-4 当前只关心两条真稀疏路径：

- `stream_llm` —— **已重写为 hardware band + sink + LSE merge**（不再走 TileLang）
- `block_sparse` —— 仍是 TileLang path-B（暂未动）
- 验证模型固定 Phi-3-mini-128k-instruct

不要回到已删除的老路线。Backup：

```text
/data/guoshiyao/zhw/MInference-NPU_backup_20260527_122449.tar.gz
```

## Runtime

```bash
cd /data/guoshiyao/zhw/MInference-NPU/MInference-NPU
source ~/ascend/cann/8.5.0/cann-8.5.0/set_env.sh
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python <cmd>
```

Local model：`/data/guoshiyao/resources/models/Phi-3-mini-128k-instruct`

`PYTHONPATH` 里的 `~/tilelang-ascend` 现在只为 `block_sparse` 需要；stream_llm 路径已不依赖 TileLang。

## 性能诊断（2026-05-27）

### 实测数字（改造前）

| ctx | stream probe | block probe | dense baseline |
|---:|---:|---:|---:|
| 4K | 22.46s | 22.26s | 0.60s |
| 8K | 30.48s | 30.18s | 0.97s |
| 16K | 49.98s | 50.98s | 2.32s |

4K 稳态（`--num-runs 2` 的 run2）：stream 4.50s / block 3.50s，**仍 7.5× 慢于 dense 0.60s**。
JIT 已排除，瓶颈在 grouped TileLang wrapper / kernel 本身。

逐层拆解：6 个 active layer × ~0.67s/layer ≈ 4s。dense 同层 attention ~0.003s。**单层稀疏 attention 比 dense 慢 ~200×**。

### 根因（三层叠加，每层都来自结构性错配）

1. **kernel 16× 算力浪费**：`tilelang_sparse_attention.py:137` 内 `padded_head_kv = max(next_pow2(h), 16)`。调用方传 `heads=1, kv_group=1` → padded=16 → `H_per_block=16`。kernel 每个 core 处理 16 个 head 槽，其中 15 个是无效计算。L1/L0C 上的 `acc_s_l0c[16, 64]`、`acc_o_l0c[16, 128]` 全被算了，结果只取 1 行。

2. **K/V 单 token gather**：`tilelang_sparse_attention.py:270-281` 内层是 `for bi_i in range(BI//2): T.copy(K[b_i, pos, g_i, :D], k_ub)`。每个 query 做 topk=1024 次单 token 散读，HBM burst 带宽吃不到。stream_llm 的 K 位置本来就是两段**连续区间**，本应 range load。

3. **Wrapper 强制 fold-into-batch**：`kv_group=1` MVP + `padded_head_kv != head_kv and kv_group != 1` 断言导致真 MHA（H 个不同 K）无法直接表达，wrapper 只能把 `B*H` 折叠到 batch 维 → 问题 1 被永远放大 16×。

三者相乘 ≈ 实测的 200× 慢。

## 已执行改动（2026-05-27 stream_llm 切 band+sink）

### 决策

stream_llm 完全弃用 TileLang，改走：

1. **Sliding window**：`torch_npu.npu_fusion_attention(sparse_mode=4, pre_tockens=n_local-1, next_tockens=0)` — Ascend 硬件 band attention，**kernel 级真稀疏**，只算 band 内的 token。
2. **Sink**：`sparse_mode=1` 跑一个 `K_len = n_init` 的极小 dense FA + bool mask（mask 形状 `[S_q, n_init]`，n_init 一般 ≤ 256，不爆炸）。
3. **LSE merge**：用两段返回的 `softmax_max` / `softmax_sum` 在线合并。

整体 FLOPs ≈ `O(S × (n_init + n_local))`，与上游 streaming-LLM 一致。

### 与已弃用 path A 的区别

- path A = `sparse_mode=1 + 大 bool mask` → 仍要做 dense QK matmul，再 mask 出 -inf。算力没省，O(S²) 显存。
- 本路径 = `sparse_mode=4 (band)` → 硬件 kernel 级真稀疏，只算 band。sink 单独走 `sparse_mode=1` 但 K_len 极小。

两者本质不同，本路径不属于 path A 复辟。

### 与已弃用 TileLang 通用稀疏的区别

- TileLang sparse_attention_fwd = 通用 per-token gather kernel，对**位置确定**的 streaming 是抽象错配。
- Hardware band 直接表达"连续区间访问"，命中硬件 burst 带宽与 sparse 算子的真稀疏实现。

### 代码改动

- `minference/ops/streaming_kernel_npu.py`
  - `_streaming_npu`：保留符号，实现换成 band + sink + LSE merge。
  - 删除 `_streaming_tilelang_npu`、`_load_sibling_module`、`_SIBLING_MODULE_CACHE`、`_TILELANG_BLOCK_SIZE`、`importlib`/`os`/`sys` 链。
  - 删除 `n_init % 64 == 0 / n_local % 64 == 0` 约束（band 可任意 int）。
  - `streaming_forward` 主路径不再 try/except TileLang，直接 `_streaming_npu`；非 NPU 仍走 PyTorch ref。
- `examples/run_hf_minimal.py`：profile 钩子从 `_streaming_tilelang_npu` 改挂到 `_streaming_npu`。
- `block_sparse_kernel_npu.py` 和 `tilelang_*.py` 未动。

### 等待验证（服务器上跑）

1. `tests/test_streaming_kernel.py` 全过（特别是 NPU vs PyTorch ref 三个用例，容差 1e-2）。
2. `tests/test_block_sparse_kernel.py` 无回归（防御性）。
3. `run_hf_minimal.py --ctx-len 4096 --attn-type minference --num-runs 2`：稳态 stream probe **应 ≤ dense 0.60s**（理论 FLOPs 比 dense 少 4×）。
4. 8K / 16K 趋势：sparse 优势随 S 增大（dense 是 O(S²)，band+sink 是 O(S × 1024) 常数）。

性能门槛：4K stream attention 一层 < 0.01s（vs 改造前 0.67s）。

## Next Step

按服务器测试结果分支：

- **数值不过** → 修 LSE merge 或 band 行/列对齐（`sparse_mode=4` 的 `pre_tockens` 语义、S_q vs S_k 对齐）。
- **数值过、速度未达标** → 看 `branch timings` 拆是 band pass 还是 sink mask 构建卡，针对性优化（sink mask 可以预算并跨层 cache、band 算子可以试 `next_tockens` / `sparse_mode` 别的取值）。
- **达标** → 进入 block_sparse 改造。

### block_sparse 待办（下一阶段）

block_sparse 的 indices 是**数据相关**（pooled QK topk），hardware band 不适用。两个候选方向：

1. **重写 TileLang kernel 支持 MHA**：去掉 `padded_head_kv != head_kv and kv_group != 1` 断言；把 K/V 改成 per-head（kv_group=H 或 heads=H 直传，不再 fold-into-batch）；内层把单 token gather 改成 per-block range load（block_sparse 的 K 位置以 block 为粒度天然连续）。工作量大，但是长期方向。
2. **短序列退 path A**：4K-16K 范围内 `sparse_mode=1 + bool mask` 显存可接受，FLOPs 比 dense 一样但少 launch 开销；≥32K 仍需 kernel。临时妥协。

方向 1 必须做，方向 2 可作过渡。stream_llm 跑通后再立决议。

## Important Files

- `minference/ops/streaming_kernel_npu.py` —— **band + sink + LSE merge**，stream_llm 新主路径
- `minference/ops/block_sparse_kernel_npu.py` —— TileLang path-B（未改）
- `minference/ops/tilelang_sparse_attention.py` —— 仅 block_sparse 用
- `minference/ops/tilelang_indices.py` —— 仅 block_sparse 用
- `minference/modules/minference_forward.py` —— grouped scheduling 不变
- `benchmarks/prepare_phi3_pathb_configs.py` —— 生成两个 dense-others probe configs
- `examples/run_hf_minimal.py` —— HF smoke runner，已更新 profile 钩子
