# Context Checkpoint

Last update: 2026-05-27（晚 — block_sparse 短序列 path A + TileLang range-load 已接入，下一步 MHA）

## Current Direction

PR-4 当前只关心两条真稀疏路径：

- `stream_llm` —— **已重写为 hardware band + sink + LSE merge，并完成 4K-64K 验证**（不再走 TileLang）
- `block_sparse` —— 4K-16K 默认先走 bool-mask NPU path A 作为过渡/对照；32K+ 仍是 TileLang path-B。TileLang path-B 已接入连续 block range-load，**下一阶段重点是去掉 fold-into-batch / padded-H 浪费**
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

## 本轮进展（2026-05-27 block_sparse staging）

- `minference/ops/block_sparse_kernel_npu.py`
  - 新增短序列调度策略：fp16 NPU block_sparse 在 `S <= 16384` 时默认优先走 `_block_sparse_npu`（`sparse_mode=1 + bool mask`）。
  - 新增环境变量 `MINFERENCE_BLOCK_SPARSE_MASK_MAX_SEQ` 控制阈值；设为 `0` 可强制短序列也先测 TileLang path-B。
  - `S > 16384` 继续优先走 `_block_sparse_tilelang_npu`，避免 32K+ O(S²) mask。
- `tests/test_block_sparse_kernel.py`
  - 新增短序列 path A 调度策略测试，覆盖默认阈值、禁用开关、非法环境变量回退。
- 验证：
  - `python -m py_compile minference/ops/block_sparse_kernel_npu.py tests/test_block_sparse_kernel.py`：通过。
  - `conda run -n flexhead-tl python -m pytest tests/test_block_sparse_kernel.py -q`：32 passed。
- 端到端 probe（Phi-3-mini-128k-instruct，block_sparse probe config，`max_new_tokens=1`，`--num-runs 2`，比较 run2）：

| ctx | dense run2 | block probe run2 | block_sparse branch | dense-others branch |
|---:|---:|---:|---:|---:|
| 4K | 0.29s | 0.32s | 0.380s / 12 calls | 0.293s / 64 calls |
| 8K | 0.69s | 0.75s | 0.503s / 12 calls | 0.653s / 64 calls |
| 16K | 2.05s | 2.24s | 0.823s / 12 calls | 2.369s / 64 calls |

结论：短序列 path A 已把 block probe 从旧 TileLang path-B 的 4K 稳态约 3.50s 拉回 dense 同量级（4K run2 0.32s vs dense 0.29s）。16K 下 block_sparse 分支本身已明显低于 dense-others 分支，但端到端仍略慢于 dense baseline，说明公共开销、dense-others 覆盖率和 HF 4D mask 构造仍是瓶颈。

注意：这只是候选方向 2 的短期 baseline/过渡，不是最终性能解。它可以减少 4K-16K 里 TileLang fold-into-batch/padded-H 的调度和 kernel 负担，但仍然是 dense QK + mask，不能解决 32K+ 的 O(S²) 内存，也不能证明真稀疏收益。

## 本轮进展（2026-05-27 TileLang range-load）

- 试探 native `torch_npu.npu_block_sparse_attention`：当前 CANN/libopapi 不可用，报错 `aclnnBlockSparseAttention... not in libopapi.so`，不能作为当前主线。
- `minference/ops/tilelang_sparse_attention.py`
  - 新增 `use_contiguous_range_load` 编译参数，默认 `False` 保持旧 sparse kernel 行为。
  - 当开关为 `True` 时，每个 `BI=block_size` 的连续 K/V block 先 range-copy 到 UB gather buffer，再写入 workspace，替代逐 token `T.copy(K[b,pos])`。
  - 注意：range-load path 只适用于完整连续 block；非整除 `S_k % block_size != 0` 时 wrapper 会回到旧 per-token gather，避免末尾 partial block 越界。
- `minference/ops/block_sparse_kernel_npu.py`
  - TileLang path-B 调用 `build_sparse_attention_qkv_fwd(..., use_contiguous_range_load=(S_k % block_size == 0))`。
- `tests/test_tilelang_sparse_attention.py`
  - 新增 `range-load-one-block` / `range-load-two-block` smoke case。
- `tests/test_block_sparse_kernel.py`
  - 新增 `test_tilelang_path_b_forced_vs_pytorch_ref`，通过 `MINFERENCE_BLOCK_SPARSE_MASK_MAX_SEQ=0` 强制短序列走 TileLang path-B，覆盖 wrapper 接入。
- 验证：
  - `python -m py_compile minference/ops/tilelang_sparse_attention.py minference/ops/block_sparse_kernel_npu.py tests/test_tilelang_sparse_attention.py tests/test_block_sparse_kernel.py`：通过。
  - `tests/test_tilelang_sparse_attention.py --case all`：8/8 PASS；range-load cases 最大误差 `9.77e-4`。
  - `python -m pytest tests/test_block_sparse_kernel.py -q`：33 passed。
- forced TileLang 4K probe（`MINFERENCE_BLOCK_SPARSE_MASK_MAX_SEQ=0`，block_sparse probe config，`--num-runs 3`）：
  - run1：17.97s（含 JIT）
  - run2：1.63s
  - run3：1.62s
  - 对比 checkpoint 中旧 TileLang 4K 稳态约 3.50s，range-load 后约 2.16× 改善。

结论：per-token HBM scatter 这一层已经被明显缓解，但 forced TileLang path-B 仍远慢于短序列 path A / dense 同量级结果（4K path A run2 0.32s，dense 0.29s）。剩余主瓶颈仍是 `heads=1, kv_group=1` 折 batch 后触发的 16× padded-H 计算浪费，以及 grouped wrapper 的调度形态。下一步不要继续微调 range-load，转向 MHA/GQA 表达或专门的 H=1 kernel。

### compact head-block 试探（失败，勿重复）

尝试给当前 `tilelang_sparse_attention.py` 增加 `min_head_block`，把 folded `heads=1` 的 `H_per_block` 从 16 降到 2/4/8：

- `H_per_block=2`：能编译，但 `T.reduce_max(acc_s_ub, m_i, dim=-1)` 要求 reduce 输出 shape 跟 `v_block=1` 匹配；修成真实 `ub_len=1` 后运行触发 AICore `VEC supports illegal configurations`。
- `H_per_block=4`：同样运行触发 AICore vector 非法配置。
- `H_per_block=8`：同样运行触发 AICore vector 非法配置。

结论：当前 C/V 双-lane pipeline 的 vector/softmax/update 路径实际依赖 `v_block >= 8`，也就是 `H_per_block >= 16`。不能靠简单缩小 `padded_head_kv` 解决 16× 浪费；必须换 kernel 结构：

- 专门 H=1/GEMV-style kernel：用 vector dot/reduce 直接算单 head，不走 `gemm_v0([H_per_block,D] x [BI,D])`。
- 或 MHA/GQA kernel：一次处理真实多 head，避免 wrapper 把 head 折进 batch 后每个 head 单独付 16-row cube 成本。

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

## stream_llm 新路径验证结果（2026-05-27）

### 关键实现经验

- `torch_npu.npu_fusion_attention(sparse_mode=4)` **必须传入 2048x2048 compressed causal mask**；`atten_mask=None` 时 `sparse_mode/pre_tockens/next_tockens` 会被忽略，实际退化为 full attention。这是本阶段最关键的坑。
- `S_q < S_k` 时，band FA 的 query row id 与 key row id 不自动按绝对位置对齐；当前实现通过在 Q 前补 dummy rows，把真实 query row 对齐到 `abs_i = S_k - S_q + i`，再切回尾部输出。
- sink pass 使用极小 `K_len=n_init` 的 `sparse_mode=1 + bool mask`，再与 band pass 做 LSE merge。该 mask 不是旧 path A 的大 SxS mask。

### 正确性

- `tests/test_streaming_kernel.py -q`：32 passed。
- `tests/test_block_sparse_kernel.py -q`：29 passed（防御性回归）。
- 64K isolated stream kernel 抽样精度：`S=65536, H=4, D=128, n_init=128, n_local=3968`，与 CPU fp32 精确定义逐行对比，抽样行最大误差 `9.77e-4`。该误差是 fp16 FA 数值误差量级，含义是 **对 streaming-LLM 稀疏定义本身的 kernel 数值误差**，不是与 dense attention 语义的误差。

### 端到端 probe 时延

配置口径：`Phi_3_mini_128k_instruct_pathb_stream_llm_aligned_dense_others.json`，每次 run 中 43 个 head 走 `stream_llm`，981 个 head 走 dense-others。branch timing 包含 attention 分支调用时间；`dense` 指 dense-others head 的 dense attention 累计时间，不是整模型 dense baseline。

| ctx | run2 | dense branch | stream_llm branch |
|---:|---:|---:|---:|
| 4K | 0.31s | 0.235s / 64 calls | 0.048s / 12 calls |
| 8K | 0.70s | 0.654s / 64 calls | 0.049s / 12 calls |
| 16K | 2.03s | 2.376s / 64 calls | 0.057s / 12 calls |
| 32K | 6.52s | 9.477s / 64 calls | 0.076s / 12 calls |
| 64K | 28.88s（2 NPU, `device_map=auto`） | 18.066s / 64 calls | 0.122s / 12 calls |

64K 平均每 head attention 分支耗时（按 2 runs 聚合统计）：

- dense-others：`18.066s / (2 * 981 heads) = 9.21ms/head`
- stream_llm：`0.122s / (2 * 43 heads) = 1.42ms/head`
- 平均每 head 加速比：`9.21 / 1.42 = 6.49×`

注意：这个 `6.49×` 是当前 probe config 下 **64K attention 分支每 head 平均耗时** 的比较。它不是端到端模型加速比，也不是全 head 都切到 stream_llm 后的预测值。4K/8K 下 stream_llm 每 head 反而受额外 launch 与 LSE merge 开销影响，不一定快于 dense；长上下文下优势才明显展开。

### 端到端 dense baseline 对比

同口径：Phi-3-mini-128k-instruct，本地权重，`max_new_tokens=1`，`--num-runs 2`，比较 run2。64K 使用 2 NPU + `device_map=auto`，其余为单 NPU。

| ctx | dense run2 | stream probe run2 | E2E speedup |
|---:|---:|---:|---:|
| 4K | 0.28s | 0.31s | 0.90× |
| 8K | 0.69s | 0.70s | 0.99× |
| 16K | 2.05s | 2.03s | 1.01× |
| 32K | 6.67s | 6.52s | 1.02× |
| 64K | 29.61s | 28.88s | 1.03× |

结论：当前 stream_llm kernel 的算子迁移已成功，但 **当前 probe config 没有实质端到端加速**。原因是仅 43 / 1024 heads 走 `stream_llm`，其余 981 heads 仍为 dense-others；端到端主要被 dense-others、HF/Phi-3 4D causal mask 构造、QKV/O proj、调度/同步等公共开销主导。下一阶段要证明项目级收益，必须扩大稀疏覆盖面（例如 block_sparse 改造或更多 heads 切 sparse），并消除/绕开 HF 端 O(S²) mask 构造。

## 已执行改动（2026-05-27 stream_llm 切 band+sink）

### 决策

stream_llm 完全弃用 TileLang，改走：

1. **Sliding window**：`torch_npu.npu_fusion_attention(sparse_mode=4, pre_tockens=n_local-1, next_tockens=0, atten_mask=compressed_2048x2048_causal_mask)` — Ascend 硬件 band attention，**kernel 级真稀疏**，只算 band 内的 token。
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
  - 新增 2048x2048 compressed causal mask cache，确保 `sparse_mode=4` 真生效。
  - 支持 `S_q < S_k` 的 band row 对齐（Q 前补 dummy rows，输出切尾部）。
  - 删除 `_streaming_tilelang_npu`、`_load_sibling_module`、`_SIBLING_MODULE_CACHE`、`_TILELANG_BLOCK_SIZE`、`importlib`/`os`/`sys` 链。
  - 删除 `n_init % 64 == 0 / n_local % 64 == 0` 约束（band 可任意 int）。
  - `streaming_forward` 主路径不再 try/except TileLang，直接 `_streaming_npu`；非 NPU 仍走 PyTorch ref。
- `examples/run_hf_minimal.py`：profile 钩子从 `_streaming_tilelang_npu` 改挂到 `_streaming_npu`。
- `block_sparse_kernel_npu.py` 和 `tilelang_*.py` 未动。

### 验证状态

- 数值：通过。
- stream_llm branch：4K-64K 均显著快于旧 TileLang streaming path。
- 端到端：当前 stream probe 没有实质 E2E 加速（64K 约 1.03x），原因是稀疏覆盖率只有 43 / 1024 heads。这个结果可以接受，说明下一阶段必须扩大稀疏覆盖面，而不是继续微调 stream_llm。

## Next Step

进入 `block_sparse` 改造。目标不是再优化 stream_llm，而是把本阶段经验迁移到数据相关稀疏路径上，扩大真实稀疏覆盖面，争取端到端收益。

### block_sparse 待办（下一阶段）

block_sparse 的 indices 是**数据相关**（pooled QK topk），hardware band 不适用。stream_llm 阶段给出的可复用经验：

- 必须避免小 H 被 padding 到 16 后做无效计算。
- 必须避免 per-token HBM scatter；block_sparse 的 key 位置以 block 为粒度天然连续，应改成 per-block range load。
- 必须减少 fold-into-batch 带来的 launch/调度和 H 维表达损失。

当前执行路线：

1. **已完成：短序列退 path A 对照**。4K-16K 范围内 `sparse_mode=1 + bool mask` 显存可接受，FLOPs 与 dense 一样但能避开当前 TileLang 16× padded-H/fold-into-batch 开销；≥32K 仍需 kernel。
2. **已完成一半：per-block range load**。
   - block_sparse 的完整 K/V block 已不再逐 token gather。
   - forced TileLang 4K 稳态从旧约 3.50s 降到 1.62s，但仍慢于 dense/path A。
3. **必须做：重写 TileLang kernel 支持 MHA/GQA 或专门 H=1 kernel**。
   - wrapper 不再把 `B*H` 折到 batch；按真实 head/group 表达 K/V 与 indices。
   - kernel 取消“最少 16 个 head 槽全算”的结构性浪费，或显式只计算有效 head。
   - 优先调查两条分支：
     - 基于 tilelang-ascend GQA 示例改 separate Q/K/V kernel，处理 `kv_group > 1` 和 small `head_kv` 的 Q/Output mask。
     - 为当前 fold-into-batch 的 `heads=1` 场景写专门 scalar-head/GEMV-style kernel，绕开 `H_per_block=16` 的 cube 浪费。不要再尝试缩小现有 C/V pipeline 的 `H_per_block`。
   - 先做 isolated kernel 单测，再接回 `block_sparse_attention`。
4. **benchmark 顺序**。
   - 先跑 block probe 4K/8K/16K，验证短序列 path A 是否至少不再比 dense 慢 7.5×。
   - 设置 `MINFERENCE_BLOCK_SPARSE_MASK_MAX_SEQ=0` 重跑同样 probe，保留 TileLang path-B 旧口径对照。
   - MHA/H=1 kernel 完成后再跑 32K/64K；短序列 path A 不作为 32K+ 方案。

方向 2（短序列 path A）已接入，只是对照 baseline。方向 1 仍是项目级收益的主线。

## Important Files

- `minference/ops/streaming_kernel_npu.py` —— **band + sink + LSE merge**，stream_llm 新主路径
- `minference/ops/block_sparse_kernel_npu.py` —— block_sparse 调度入口；4K-16K 默认 path A 对照，32K+ TileLang path-B
- `minference/ops/tilelang_sparse_attention.py` —— 仅 block_sparse 用
- `minference/ops/tilelang_indices.py` —— 仅 block_sparse 用
- `minference/modules/minference_forward.py` —— grouped scheduling 不变
- `benchmarks/prepare_phi3_pathb_configs.py` —— 生成两个 dense-others probe configs
- `examples/run_hf_minimal.py` —— HF smoke runner，已更新 profile 钩子
