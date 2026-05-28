# Context Checkpoint

Last update: 2026-05-28（深夜 — Tier 1 MH TileLang kernel；wrapper 取消 fold-into-batch；E2E quality 横扫降级，回到算法迁移 + 速度路线）

## Current Direction

PR-4 当前只关心两条真稀疏路径：

- `stream_llm` —— **已重写为 hardware band + sink + LSE merge，并完成 4K-64K 验证**（不再走 TileLang）
- `block_sparse` —— 4K-16K 默认先走 bool-mask NPU path A 作为过渡/对照；32K+ 仍是 TileLang path-B。TileLang path-B 已接入连续 block range-load、H=1 query-block kernel、block-index H=1 kernel，并已修复 `device_map=auto` 多卡 TileLang cache/device 绑定问题。dense-others 已改为 hardware causal band。**token-level indices 展开/OOM 已解除；当前最好端到端点是 layers8-13 全 heads + topk1，在 32K 和 64K 多卡 run2 都小幅/明显超过 dense baseline**
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

## 本轮进展（2026-05-28 深夜 — Tier 1 MH TileLang kernel + 路线收回）

### 方向纠正（重要）

E2E quality 横扫被降级。原因：

- 项目使命是 **block_sparse 算法迁移 + 速度 + 合理精度**，不是端到端寻找最佳 layer/topk。
- 精度通过单元测试（NPU vs PyTorch ref）守底，已有 39 passed。
- 端到端精度下降在"所有 head 都强转 block_sparse"的配置上**必然发生**，因为 block_sparse 不是大多数 head 的最佳 pattern。这是 best_pattern 上游的事。
- `--save-output` / `--compare-to` / `--offline-compare` / `--prompt-file` 等工具保留，仅作可选 sanity check，不再作里程碑验收。

实测发现的回归（用 Alice in Wonderland 真实文本 prompt，下面这组数据催生路线收回）：

| metric | layers8-13 topk1 / 32K real prompt |
|---|---:|
| dense run2 | 6.50s |
| sparse run2 | 6.60s（输 1.5%） |
| token greedy match | 4/32（12.5%） |
| ref-mass covered | 0.1755 |

这说明 layers8-13 全 heads topk1 之前的 "32K 小幅赢 dense" 数字是 prompt 退化导致的虚假胜利，不是稳态。算法迁移路线下不应纠结这种配置，应该让 best_pattern 自然分配 sparse / dense 比例。

### Tier 1 MH TileLang kernel（本轮主代码改动）

**动机**：当前 wrapper 把 ``B×H`` 折到 batch 维 + ``head_chunk=8`` 切分，导致 layers8-13 single layer 一次 sparse forward 触发 4 次 kernel call。每次 call 携带 Python 端 reshape / contiguous / JIT lookup / NPU stream sync。32K 实测 48 calls × 346ms = 16.6s，其中 wrapper + launch 开销有相当部分。

**改动**：

- `minference/ops/tilelang_sparse_attention_h1.py`
  - 新增 `build_sparse_attention_mh_block_index_fwd(...)`。
  - 输入 `q/k/v [B, S, H, D]`、`block_indices [B, n_q_blocks, H, topk_blocks]`，输出 `[B, S, H, D]`。
  - 工作单元分解 `pid = b*H*q_tiles + h*q_tiles + tile_i`；每个 work item 仍处理 16 query token × 1 head，cube/vector pipeline 与 H=1 block-index 完全一致，仅索引带上 H 维。
  - cache key 带 `mh_block_index` 前缀避免与 H=1 cache 冲突。
- `minference/ops/block_sparse_kernel_npu.py`
  - `_block_sparse_tilelang_npu` 默认先走 MH path。新增 `_should_use_tilelang_mh_block_index()`：依赖 H=1 block-index 安全边界 + 没被 `MINFERENCE_BLOCK_SPARSE_TILELANG_MH=0` 显式关闭。
  - MH path：`q/k/v` 直接 BHSD → BSHD permute 一次，不再 fold-into-batch；indices `[B, H, n_q_blocks, topk]` → `[B, n_q_blocks, H, topk]`。
  - 回退路径：MH 不启用时走原 H=1 fold-into-batch，再不行走旧 padded-H kernel。
- `tests/test_tilelang_sparse_attention_h1.py`
  - 新增 `mh-h2-one-block`、`mh-h4-two-block`、`mh-h8-one-block` isolated case。每个 head 选不同 K block 子集，确保 MH path 正确处理 per-head sparse pattern。
- `tests/test_block_sparse_kernel.py`
  - 新增 3 个策略测试：MH 默认启用、`MINFERENCE_BLOCK_SPARSE_TILELANG_MH=0` 强制回退、关 block-index 时 MH 也不启用（依赖关系）。
- `benchmarks/bench_tilelang_h1_sparse_attention.py`
  - 新增 `--mh-heads` / `--mh-head-chunks` 参数。`run_mh_vs_h1_case` 直接对比 MH 一次 launch vs H=1 fold-into-batch（含 head_chunk 多次 launch）。

**Tier 1 期望收益**：

- 单层 sparse forward kernel-call 数从 4 降到 1（layers8-13 topk1 配置下，per layer per run），48 → 12 calls 数量级。
- 估算 1-3s / 32K run 的 Python wrapper 开销节省。
- cube/vector 实际计算时间不变（每个 work item 内部完全一致）。
- 若 wrapper 开销不是主导项，Tier 1 收益小；这就需要 Tier 2（cube 内跨 head 批处理）。

**没动**：

- `_block_sparse_head_chunk_size()` 默认仍 8。MH path 下用 `MINFERENCE_BLOCK_SPARSE_HEAD_CHUNK=0` 单独打开"一次 launch 全 H"，避免改默认值影响别处。
- stream_llm 路径完全没动。
- best_pattern 调度逻辑没动。

### 验证

- `python -m py_compile minference/ops/tilelang_sparse_attention_h1.py minference/ops/block_sparse_kernel_npu.py tests/test_tilelang_sparse_attention_h1.py tests/test_block_sparse_kernel.py benchmarks/bench_tilelang_h1_sparse_attention.py`：全部通过。
- 服务器侧待跑（命令见下）。

### 服务器端待跑命令

按以下顺序执行；任何一步失败就回退 `MINFERENCE_BLOCK_SPARSE_TILELANG_MH=0` 用旧路径。

```bash
# 1) isolated 数值正确性（MH path）
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python \
  tests/test_tilelang_sparse_attention_h1.py --case all

# 2) wrapper 单测回归（含 MH 策略测试）
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python \
  -m pytest tests/test_block_sparse_kernel.py -q

# 3) MH vs H=1 fold-into-batch isolated microbenchmark
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python \
  benchmarks/bench_tilelang_h1_sparse_attention.py \
  --seq-lens 4096 16384 32768 \
  --topk-blocks 1 4 16 \
  --dim 128 \
  --mh-heads 8 32 \
  --mh-head-chunks 0 8

# 4) HF 端到端：layers8-13 topk1 32K，A/B MH on vs off
#    MH on (默认)
NAME=mh_on_sparse_32k_layers8_13_topk1
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python \
  examples/run_hf_minimal.py \
  --attn-type minference \
  --config-path minference/configs/Phi_3_mini_128k_instruct_pathb_block_sparse_layers8_13_all_heads_topk1_latency.json \
  --ctx-len 32768 --max-new-tokens 1 --num-runs 2 --empty-cache-between-runs \
  --device-map npu:0 --profile-branches \
  --run-name $NAME

#    MH off（强制回退到 H=1 fold-into-batch，做对照）
NAME=mh_off_sparse_32k_layers8_13_topk1
MINFERENCE_BLOCK_SPARSE_TILELANG_MH=0 \
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python \
  examples/run_hf_minimal.py \
  --attn-type minference \
  --config-path minference/configs/Phi_3_mini_128k_instruct_pathb_block_sparse_layers8_13_all_heads_topk1_latency.json \
  --ctx-len 32768 --max-new-tokens 1 --num-runs 2 --empty-cache-between-runs \
  --device-map npu:0 --profile-branches \
  --run-name $NAME

# 5) MH + head_chunk=0（一次 launch 全 32 heads）
NAME=mh_on_chunk0_sparse_32k_layers8_13_topk1
MINFERENCE_BLOCK_SPARSE_HEAD_CHUNK=0 \
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python \
  examples/run_hf_minimal.py \
  --attn-type minference \
  --config-path minference/configs/Phi_3_mini_128k_instruct_pathb_block_sparse_layers8_13_all_heads_topk1_latency.json \
  --ctx-len 32768 --max-new-tokens 1 --num-runs 2 --empty-cache-between-runs \
  --device-map npu:0 --profile-branches \
  --run-name $NAME
```

对比目标：第 4/5 步的 `block_sparse: Xs over N calls` 中 `N` 应该从 48（chunk=8 + H=1 fold）降到 12（MH + chunk=0），平均每 call 时延应该相当或略升，但总 `X` 应下降。如果总 X 没降，说明 wrapper 开销不是主导项，需要进 Tier 2。

## 本轮进展（2026-05-28 夜 — 横扫 tooling + quality probe）

目标：把 checkpoint 里仍未完成的两条遗留收尾。`device_map=auto` 多卡问题
本周已修；剩两件是 layer range / topk 横扫和 layers8-13 topk1 的输出质量验证。
本轮不动 kernel，纯 tooling。

### 新增/扩展

- `benchmarks/prepare_phi3_pathb_configs.py`
  - 保留原有固定 layers8-13 配置以避免回归。
  - 新增 `--extra-layer-ranges`（如 `"7-14,8-12,9-12,8-14"`）与
    `--extra-topks`（如 `"1,2,4"`）做 cartesian。
  - 输出统一命名 `Phi_3_mini_128k_instruct_pathb_block_sparse_layers{lo}_{hi}_all_heads_topk{k}_latency.json`。
- `examples/run_hf_minimal.py`
  - 新增 `--save-output PATH`：把 generate 的 `output_scores=True` 拿到的
    per-step top-K logits + 完整 token ids dump 成 JSON。
  - 新增 `--compare-to PATH`：读 reference JSON 与当前 run 对比，给出：
    1. token greedy 完全匹配率 + longest matching prefix + first divergence 步号。
    2. per-step top-1 命中率、top-5 jaccard 重合率。
    3. 基于 union(top-K) 的近似 KL(ref || cur)，单位 nats。
       某步出现 ref top-K token 不在 cur top-K 时该步标记 +inf 并跳过；
       全部 +inf 则提示 "分布偏移很大"。
  - 新增 `--quality-top-k`（默认 32）控制保留的 top-K 大小。
  - 新增 `--log-file PATH` / `--run-name NAME`：
    - 把 stdout tee 到该日志文件，方便 server 上跑完后 commit/push 给开发机看。
    - 裸文件名（无目录分隔）自动落到 `benchmarks/results/runs/`。
    - `--run-name foo` 等价于 `--log-file foo.log`。
- `benchmarks/results/runs/.gitkeep`：保留目录跟踪。该目录里的 `.log` / `.json`
  允许定期手动清理，但默认 commit 进 git，方便在开发机 `git pull` 后直接看。

### 推荐横扫矩阵

先做 4 组 quality reference（一次性 dense baseline，后续所有 sparse 都拿它对照）：

- ctx ∈ {4096, 16384, 32768, 65536}
- attn_type = dense

再扫 layer ranges × topks：

- layer ranges: `7-14,8-13,8-12,9-12,8-14,9-14,10-13`
- topks: `1, 2, 4`
- ctx: 优先 32K（已知小幅赢 dense）和 64K（已知大幅赢 dense）

每组都在同一 prompt（`run_hf_minimal.py` 默认 base prompt × ctx_len）上跑
`--max-new-tokens 32 --num-runs 2`，run2 当 latency，scores 当 quality。
然后 `--compare-to` 对应 ctx 的 dense reference，把 token match% / KL 写日志。

### 验证

- `python -m py_compile benchmarks/prepare_phi3_pathb_configs.py examples/run_hf_minimal.py`：通过。
- `python -c "...; _parse_layer_range('8-13')"` 等 helper smoke：通过。
- `python -c "...; _resolve_run_path(Path('dense_32k.json'))"`：返回
  `benchmarks/results/runs/dense_32k.json`。

### 服务器侧运行命令（直接 copy）

环境变量略，按 Runtime section。下面所有命令都写 `--run-name <NAME>`，
日志会落到 `benchmarks/results/runs/<NAME>.log`；commit/push 后我能直接读。

1. 生成新 configs（一次）：

```bash
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python \
  benchmarks/prepare_phi3_pathb_configs.py \
  --extra-layer-ranges "7-14,8-12,9-12,8-14,9-14,10-13" \
  --extra-topks "1,2,4"
```

2. 4 个 ctx 的 dense reference（每次保存 quality JSON，方便后续 compare）：

```bash
for CTX in 4096 16384 32768 65536; do
  DM=npu:0; [ "$CTX" = "65536" ] && DM=auto
  NAME=dense_ref_ctx${CTX}
  PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python \
    examples/run_hf_minimal.py \
    --attn-type dense --ctx-len $CTX --max-new-tokens 32 --num-runs 2 \
    --empty-cache-between-runs \
    --device-map "$DM" \
    --save-output ${NAME}.json --run-name $NAME
done
```

3. 32K layers8-13 topk1 quality + latency 对照（这是目前最佳点的质量验证）：

```bash
NAME=sparse_32k_layers8_13_topk1
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python \
  examples/run_hf_minimal.py \
  --attn-type minference \
  --config-path minference/configs/Phi_3_mini_128k_instruct_pathb_block_sparse_layers8_13_all_heads_topk1_latency.json \
  --ctx-len 32768 --max-new-tokens 32 --num-runs 2 --empty-cache-between-runs \
  --device-map npu:0 \
  --profile-branches \
  --save-output ${NAME}.json \
  --compare-to dense_ref_ctx32768.json \
  --run-name $NAME
```

4. 64K 多卡 layers8-13 topk1 quality + latency 对照：

```bash
NAME=sparse_64k_layers8_13_topk1_auto
PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python \
  examples/run_hf_minimal.py \
  --attn-type minference \
  --config-path minference/configs/Phi_3_mini_128k_instruct_pathb_block_sparse_layers8_13_all_heads_topk1_latency.json \
  --ctx-len 65536 --max-new-tokens 32 --num-runs 2 --empty-cache-between-runs \
  --device-map auto \
  --profile-branches \
  --save-output ${NAME}.json \
  --compare-to dense_ref_ctx65536.json \
  --run-name $NAME
```

5. layer range 横扫（按 32K 先一遍，然后挑一两个有希望的扩到 64K）：

```bash
for LR in 7_14 8_13 8_12 9_12 8_14 9_14 10_13; do
  for K in 1 2 4; do
    NAME=sparse_32k_layers${LR}_topk${K}
    CFG=minference/configs/Phi_3_mini_128k_instruct_pathb_block_sparse_layers${LR}_all_heads_topk${K}_latency.json
    PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python \
      examples/run_hf_minimal.py \
      --attn-type minference --config-path $CFG \
      --ctx-len 32768 --max-new-tokens 32 --num-runs 2 --empty-cache-between-runs \
      --device-map npu:0 --profile-branches \
      --save-output ${NAME}.json \
      --compare-to dense_ref_ctx32768.json \
      --run-name $NAME
  done
done
```

跑完后只要 `git add benchmarks/results/runs/ && git commit -m 'sweep logs' && git push`，
我在开发机 `git pull` 就能直接读 `.log` / `.json`，不用你贴大段输出。

## 本轮进展（2026-05-28 多卡修复 + 上游算子对照）

### device_map=auto 多卡修复

问题：

- 64K + layers8-13 all heads topk1 在单卡 `npu:0` 可跑，run2 `20.00s`。
- 同一配置在 `device_map=auto` 下触发 TileLang AICore：`MTE instruction DDR address out of range`。
- 隔离 wrapper H=1/2/4/8/16/32 64K topk1 全部可跑，H=32 `68.51ms`，说明不是 kernel 本体或 folded-head batch 的基本边界。

根因判断：

- TileLang callable 的 Python cache 只按 shape/topk 等参数缓存，没有按 NPU device 缓存。
- `device_map=auto` 下不同层在不同 NPU 上执行，可能复用第一次在 `npu:0` 编译的 callable 去跑 `npu:1` tensor。

代码改动：

- `minference/ops/block_sparse_kernel_npu.py`
  - 新增 `_set_current_npu_device_for(q)`，TileLang JIT/call 前切到 `q.device`。
  - `_block_sparse_tilelang_npu` 向 TileLang builder 传入 `cache_device="npu:{idx}"`。
- `minference/ops/tilelang_sparse_attention_h1.py`
  - `build_sparse_attention_h1_block_fwd` 和 `build_sparse_attention_h1_block_index_fwd` 的 cache key 纳入 `cache_device`。
- `minference/ops/tilelang_sparse_attention.py`
  - 旧 token-index fallback kernel cache key 也纳入 `cache_device`。

验证：

- `python -m py_compile minference/ops/block_sparse_kernel_npu.py minference/ops/tilelang_sparse_attention_h1.py minference/ops/tilelang_sparse_attention.py`：通过。
- `PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python -m pytest tests/test_block_sparse_kernel.py -q`：`39 passed`。
- `PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python tests/test_tilelang_sparse_attention_h1.py --case all`：PASS。

64K 多卡 E2E 结果（Phi-3-mini-128k-instruct，`device_map=auto`，`max_new_tokens=1`，`--empty-cache-between-runs`）：

| config | run1 | run2 | notes |
|---|---:|---:|---|
| dense baseline | 23.14s | 22.41s | default dense path |
| layers8-13 all heads topk1 | 51.01s | 19.48s | 首轮含 TileLang JIT；run2 稳态赢 dense |

结论：

- 多卡问题已修复：64K `device_map=auto` 不再 AICore。
- 64K 多卡稳态 sparse `19.48s` vs dense `22.41s`，约 13.1% 加速。
- 这是比 32K 更有说服力的端到端收益点，也符合长上下文越长 sparse 越该有收益的预期。

### 对照 MInference 原始 block_sparse 算子

参考上游 `block_sparse_flash_attention.py` 后发现两点：

- 上游会 sort top-k block indices；当前 NPU path 已改为 `.indices.sort(dim=-1).values`。
- 上游 kernel 对早期 query block 使用 `block_count = min(q_block + 1, topk)`，避免计算不可见未来 block。

尝试迁移第二点到 TileLang block-index kernel：

- C/V 两侧都只处理 `i_i <= q_block_i` 的 block。
- 数值 smoke PASS。
- 但 wrapper benchmark 变慢：
  - 32K topk4：`4.52ms -> 4.73ms`
  - 32K topk16：`15.74ms -> 15.92ms`

结论：

- 在当前 TileLang/Ascend pipeline 下，动态分支和跨核同步损耗超过了跳过早期 future block 的收益。
- 已回退动态 block_count 剪枝，仅保留 indices sort 和 device-aware cache。
- 后续如果要复刻上游这部分收益，需要更深地重排 kernel pipeline，而不是简单加动态 if。

## 本轮进展（2026-05-28 low-topk 覆盖点搜索）

目标：从“all-head 太慢、probe 覆盖太少”之间找中间 sparse 覆盖点，直接服务端到端 latency。

新增/纳入生成脚本：

- `benchmarks/prepare_phi3_pathb_configs.py`
  - 新增 layers8-13 全 heads block_sparse latency configs：
    - `Phi_3_mini_128k_instruct_pathb_block_sparse_layers8_13_all_heads_topk1_latency.json`
    - `Phi_3_mini_128k_instruct_pathb_block_sparse_layers8_13_all_heads_topk2_latency.json`
    - `Phi_3_mini_128k_instruct_pathb_block_sparse_layers8_13_all_heads_topk4_latency.json`
  - 这 3 个配置只把当前 probe 涉及的 8-13 层整层 32 heads 改为 block_sparse，其余层保持 dense。
  - `MINFERENCE_BLOCK_SPARSE_HEAD_CHUNK=0` 下运行，依赖 block-index kernel 避免 all-head indices OOM。
- `minference/ops/block_sparse_kernel_npu.py`
  - 新增短序列 bool-mask path A 的 head 数保护：`MINFERENCE_BLOCK_SPARSE_MASK_MAX_HEADS`，默认 16。
  - 原因：16K + layers8-13 all-head 配置在 `H=32` 时会尝试构造巨大 bool mask，反复 OOM 后才 fallback TileLang，导致 run2 变成 `95.97s`。
  - 新策略：`S <= 16384` 仍可走 path A，但只有 `num_heads <= 16` 的小 head group 才默认走 mask；更大的 group 直接走 TileLang path-B。
- `benchmarks/probe_block_sparse_heads.py`
  - 新增隔离 head-count probe，用于定位 64K HF AICore 是否来自 kernel 本体。

32K E2E 结果（Phi-3-mini-128k-instruct，`max_new_tokens=1`，比较 run2，串行单进程跑）：

| config | sparse heads | topk_blocks | run2 | branch timings |
|---|---:|---:|---:|---|
| dense baseline（复跑） | 0 | n/a | 3.87s | n/a |
| probe + dense-others | 43 | 16 | 4.47s | block_sparse `16.511s/18 calls`, dense `4.273s/64 calls` |
| layers8-13 all heads | 192 | 4 | 4.29s | block_sparse `18.092s/12 calls`, dense `3.669s/52 calls` |
| layers8-13 all heads | 192 | 2 | 3.93s | block_sparse `16.792s/12 calls`, dense `3.675s/52 calls` |
| layers8-13 all heads | 192 | 1 | 3.74s | block_sparse `16.793s/12 calls`, dense `3.667s/52 calls` |

结论：

- 这是当前第一次在 32K 端到端 run2 上超过 dense baseline：`3.74s` vs dense `3.87s`，约 3.4%。
- topk 从 2 降到 1 后，branch profiler 的 `block_sparse` 总时长几乎没变，但 run2 下降明显；说明 profiler 包含两轮/JIT/同步和重复 run 混合，不能只看 branch 累计时间判断稳态。
- 真正有用的覆盖策略不是 all-head，也不是 43-head probe，而是“去掉部分 dense 层调用 + 低 topk 控制 sparse 计算量”。
- 下一步应该围绕 layers8-13 topk1 做横向验证：
  1. 测 layers8-13 topk1 的输出/质量风险；topk1 可能牺牲注意力覆盖，当前只证明 latency。
  2. 尝试更窄/更宽 layer range，例如 layers9-12、8-12、8-14，找 latency/质量折中。
  3. 处理 64K + `device_map=auto` 的 TileLang AICore 问题；单卡可跑，auto 多卡不可跑。

横向验证（新增）：

| ctx | config | run2 | comparison / notes |
|---:|---|---:|---|
| 16K | layers8-13 all heads topk1，head guard 前 | 95.97s | H=32 bool-mask path A 反复 OOM fallback，负例 |
| 16K | layers8-13 all heads topk1，head guard 后 | 1.40s | 不再触发 path A OOM |
| 16K | dense baseline 复跑 | 1.35s | sparse 略慢，16K 不建议启用该配置 |
| 64K | layers8-13 all heads topk1，`device_map=auto` | fail | AICore MTE out-of-range，head chunk=0 和默认 chunk=8 均失败 |
| 64K | isolated wrapper H=1/2/4/8/16/32 topk1 | pass | H=32 isolated `68.51ms`，说明 kernel 本体和 folded-head batch 可跑 |
| 64K | layers8-13 all heads topk1，single `npu:0` | 20.00s | `--num-runs 2 --empty-cache-between-runs`，run2 可跑 |

当前策略判断：

- 16K：dense 仍更快，且 path A 需要 head guard 防灾。
- 32K：layers8-13 all heads topk1 是当前最佳点，run2 `3.74s`，小幅赢 dense `3.87s`。
- 64K：单卡 topk1 可跑到 run2 `20.00s`，但 `device_map=auto` 会触发 TileLang AICore，暂不能作为多卡长上下文方案。

## 本轮进展（2026-05-28 block-index H=1 kernel + causal fast-path）

目标：消除 block_sparse path-B wrapper 里把 block indices 展开成 token-level indices 的大中间张量，并尽量让这个容量修复不拖慢 kernel。

```text
old: [B*H, n_q_blocks, topk_blocks] -> [B*H, S, topk_blocks * 64]
new: kernel 直接接收 [B*H, n_q_blocks, 1, topk_blocks]
```

代码改动：

- `minference/ops/tilelang_sparse_attention_h1.py`
  - 新增 `build_sparse_attention_h1_block_index_fwd(...)`。
  - 输入 `block_indices [B, n_q_blocks, 1, topk_blocks]`，kernel 内部按 `block_idx * block_I` 连续 range-load K/V。
  - 追加 causal fast-path：更早的 K block 整块可见，直接复用 cube 得分；只有对角/部分可见 block 才生成 `BI=64` 个 token position 并做逐 token causal compare。
- `minference/ops/block_sparse_kernel_npu.py`
  - H=1 安全边界内默认优先走 block-index kernel，避免调用 `block_indices_to_tilelang`。
  - 新增回退开关：`MINFERENCE_BLOCK_SPARSE_TILELANG_BLOCK_INDEX=0` 可回旧 token-index H=1 路径。
  - 原 `MINFERENCE_BLOCK_SPARSE_TILELANG_H1=0` 仍可回旧 padded-H kernel。
- `tests/test_tilelang_sparse_attention_h1.py`
  - 新增 block-index one-block / two-block smoke。
- `benchmarks/bench_tilelang_h1_sparse_attention.py`
  - 新增 token-index H=1 vs block-index H=1 kernel 对照。
- `tests/test_block_sparse_kernel.py`
  - 新增 block-index 默认/禁用策略测试。

验证：

- `python -m py_compile minference/ops/tilelang_sparse_attention_h1.py`：通过。
- `PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python -m pytest tests/test_block_sparse_kernel.py -q`：`37 passed`（fast-path 前；策略测试已覆盖）。
- `PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python tests/test_tilelang_sparse_attention_h1.py --case all`：4/4 PASS。
  - block-index one-block / two-block 均 `max_abs_diff=9.7656e-04`。

isolated kernel benchmark（S=4096, D=128）：

| topk_blocks | token-index H=1 | block-index H=1 before fast-path | block-index H=1 after fast-path |
|---:|---:|---:|---:|
| 4 | 1.194 ms | 1.254 ms | 1.146 ms |
| 16 | 2.389 ms | 2.945 ms | 2.145 ms |

wrapper benchmark（H=1, S=32768, D=128）：

| topk_blocks | token-index mean / peak alloc | block-index before fast-path | block-index after fast-path |
|---:|---:|---:|---:|
| 4 | 5.55 ms / 144 MB | 6.02 ms / 41 MB | 4.52 ms / 41 MB |
| 16 | 20.62 ms / 480 MB | 22.07 ms / 41 MB | 15.74 ms / 41 MB |

端到端 32K 复测（Phi-3-mini-128k-instruct，`max_new_tokens=1`，比较 run2）：

| config | before block-index | block-index before fast-path | block-index after fast-path | notes |
|---|---:|---:|---:|---|
| probe + dense-others | 4.65s | 4.71s | 4.47s | dense baseline 仍是 3.88s |
| all heads topk=4, no chunk | no-chunk 之前不可行 | 7.47s | 6.17s | 仍慢于 dense |
| all heads topk=16, no chunk | OOM | 23.57s | 未复跑 E2E | wrapper topk16 已从 22.07ms 降到 15.74ms |

重要结论：

- block-index kernel 已解决 all-head/no-chunk 的 token-index OOM；topk16 no-chunk 能跑通。
- causal fast-path 让 block-index 不再只是容量修复，也带来了真实 path-B 时延下降。
- 但端到端仍未赢 dense baseline：32K probe `4.47s` vs dense `3.88s`；all-head topk4 `6.17s` 仍过慢。
- 注意：Phi-3 E2E 不能并行跑多个进程；一次并行 probe + all-head 复测在 MLP 处 OOM。后续 E2E 必须串行。

下一步建议：

- 不要再只扩大 sparse 覆盖率；all-head 会把 block_sparse 计算量拉满。
- 更有希望的方向是找 sparse 覆盖点：只把 dense-others 中最贵/最长的部分 heads 转为 low-topk block_sparse，而不是 1024/1024 heads 全覆盖。
- kernel 侧下一步是多 head TileLang launch 或按 head group 融合，减少 64 layer-call × 多 head 的调度/折叠成本。

## 本轮进展（2026-05-27 block_sparse staging）

### E2E 继续优化：跳过 HF 4D mask + dense-others hardware band（2026-05-28）

代码改动：

- `minference/patch.py`
  - patch Phi-3 modeling module 里的 `_prepare_4d_causal_attention_mask`。
  - 当 2D `attention_mask` 无 padding（全 1）时直接返回 `None`，避免 HF 在进入 layer loop 前构造 `[B,1,S,S]` 4D causal mask。
  - 若检测到真实 padding，则回退 HF 原 helper，保证语义不乱。
- `minference/backend_npu/attention.py`
  - `dense_attention(..., causal=True)` 在 `S_q == S_k` 且 `S <= MINFERENCE_DENSE_BAND_MAX_SEQ` 时改走 `npu_fusion_attention(sparse_mode=4, pre_tockens=S-1, next_tockens=0)`。
  - 这用 Ascend hardware causal band 表达 full causal prefill，不再每次构造 `SxS` bool causal mask。
  - 默认 `MINFERENCE_DENSE_BAND_MAX_SEQ=32768`；64K full causal band 曾触发 AICore 异常，暂时保守回退旧 masked path。
- `tests/test_dense_attention_band.py`
  - 新增 dense-band standalone smoke，验证 `S=128/512/1024` 与 CPU fp32 causal reference 的最大误差均为 `9.7656e-04`。
- `examples/run_hf_minimal.py`
  - 保留上一轮新增的 `--empty-cache-between-runs`，用于长上下文重复 run 调试。

验证：

- `python -m py_compile minference/patch.py minference/backend_npu/attention.py examples/run_hf_minimal.py tests/test_dense_attention_band.py`：通过。
- `PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python tests/test_dense_attention_band.py`：通过。

E2E probe（Phi-3-mini-128k-instruct，block_sparse probe config，默认短序列 path A，`--num-runs 2`，比较 run2）：

| ctx | sparse probe run2 | current dense run2 | old dense baseline in checkpoint | notes |
|---:|---:|---:|---:|---|
| 4K | 0.29s | TBD | 0.29s | sparse probe 回到 dense 同量级 |
| 8K | 0.60s | TBD | 0.69s | dense-others band 化后明显下降 |
| 16K | 1.58s | 1.36s | 2.05s | sparse probe 也变快，但当前 dense baseline 更快 |
| 32K | 4.65s | 3.88s | 6.67s | 32K sparse probe 从上一轮 7.23s 降到 4.65s，但仍慢于新 dense baseline |

分支计时变化：

- 32K block_sparse probe：
  - dense-others branch 从上一轮 `9.399s / 64 calls` 降到 `4.266s / 64 calls`。
  - 说明 hardware causal band 正在实质降低端到端公共 dense 分支开销。
- 16K block_sparse probe：
  - branch timings: `block_sparse=0.811s / 12 calls`，`dense=1.089s / 64 calls`。

64K 注意：

- dense-band full causal 在 64K (`pre_tockens=65535`) 的 HF run 中触发过 AICore 异常；因此默认阈值限制在 32768。
- 64K 端到端仍需要后续单独处理；目前不能把 dense-band 阈值贸然拉到 64K。

### sparse 覆盖率实验：all-head block_sparse 暂不可作为当前端到端方案

为了验证“提高 sparse 覆盖率是否直接带来 E2E 收益”，新增/生成 latency configs：

- `benchmarks/prepare_phi3_pathb_configs.py`
  - 新增输出 `Phi_3_mini_128k_instruct_pathb_block_sparse_all_heads_latency.json`（1024/1024 heads 全部 block_sparse，topk_blocks=16）。
- 手动生成：
  - `Phi_3_mini_128k_instruct_pathb_block_sparse_all_heads_topk4_latency.json`（全 heads，topk_blocks=4）。
- `minference/modules/minference_forward.py`
  - 新增 `MINFERENCE_BLOCK_SPARSE_HEAD_CHUNK`，默认 8，限制 block_sparse group 每次处理的 head 数，避免全 heads 时 token-level indices 一次性展开 OOM。

实验结果（32K，全 heads）：

| config | result |
|---|---:|
| all heads, topk=16, no chunk | 第二轮 OOM：token-level indices 约 `32 * 32768 * 1024 * int32 ~= 4GiB`/layer-call |
| all heads, topk=16, chunk=8 | 能跑，run2 `22.73s`，过慢 |
| all heads, topk=4, chunk=8 | 能跑，run2 `7.31s`，仍慢于 new dense baseline `3.88s` |

结论：

- 当前 wrapper 把 block indices 展开成 token-level indices `[B*H, S, topk_blocks*64]`，这是扩大 sparse 覆盖率后的主要内存和调度瓶颈。
- 简单提高 sparse 覆盖率不够；全 heads 会被 indices 展开和 chunk 调度吃掉收益。
- 下一步应写 **block-index H=1 kernel**：kernel 直接接收 block indices `[B*H, n_q_blocks, topk_blocks]` 或 `[B*H, S/block, topk_blocks]`，内部按 block range-load K/V，不再构造 `[B*H, S, topk_blocks*64]` token indices。
- 这一步比继续微调现有 H=1 token-index kernel更接近端到端目标。

### H=1 / query-block TileLang kernel 第一阶段（isolated，已跑通）

已选择方案 A：先写专门的 H=1 kernel，绕开当前 `heads=1` 仍按 `H_per_block=16` padded-head 计算的问题。

本阶段没有接入 `block_sparse_attention` 主路径，只新增 isolated kernel 和 smoke 测试：

- `minference/ops/tilelang_sparse_attention_h1.py`
  - 新增 `build_sparse_attention_h1_block_fwd(...)`。
  - 输入仍是 folded-head 后的 BSHD/BSGD 形态：`q [B,S_q,1,D]`、`k/v [B,S_k,1,D]`、`indices [B,S_q,1,topk]`。
  - kernel 每个 work item 处理 `block_M=16` 个 query token × 1 个真实 head，而不是旧 kernel 的 1 个 query token × 16 个 padded head 槽。
  - 这样仍使用 `gemm_v0([16,D] x [64,D])` 和 `gemm_v0([16,64] x [64,D])`，但 16 行全部是真实 query token，不再是 15 行空 head。
  - V 侧 softmax 仍按 `v_block = block_M // 2 = 8` 行运行，避开之前尝试 `H_per_block=2/4/8` 时触发的 AICore vector illegal config。
  - 第一版约束：fp16、causal、`kv_group=1/head=1`、`block_M` 至少 16、运行时 `S_q` 暂按 `block_M` 整除处理；indices 预期是 block_sparse 形态，即同一个 query block 内行共享 K block list。
- `tests/test_tilelang_sparse_attention_h1.py`
  - 新增 isolated smoke：`one-block` / `two-block`。
  - reference 复用现有 `sparse_attention_qkv_reference`。

验证：

- `python -m py_compile minference/ops/tilelang_sparse_attention_h1.py tests/test_tilelang_sparse_attention_h1.py`：通过。
- `PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python tests/test_tilelang_sparse_attention_h1.py --case all`：通过。
  - `one-block`：`max_abs_diff=9.7656e-04`，PASS。
  - `two-block`：`max_abs_diff=9.7656e-04`，PASS。
- 回归确认旧 H=1 padded-head kernel smoke 仍通过：
  - `tests/test_tilelang_sparse_attention.py --case h1-one-block`：PASS，`max_abs_diff=9.7656e-04`。

踩坑记录：

- 新 TileLang kernel 文件不能使用 `from __future__ import annotations`。否则 `T.prim_func` 的参数 annotation 会被 Python 延迟成字符串，TVM parser 报 `expected Object but got str`。该问题已修复。

结论：

- 方案 A 的第一块已经证明：可以用 query-token 维度填满 16 行，绕开 padded-head 浪费，同时保留 Ascend 友好的 `v_block=8` softmax 形态。

### H=1 / query-block TileLang kernel 第二阶段（已接入 path-B）

代码改动：

- `minference/ops/block_sparse_kernel_npu.py`
  - `_block_sparse_tilelang_npu` 在 folded `B*H` 后默认调用 `tilelang_sparse_attention_h1.build_sparse_attention_h1_block_fwd(...)`。
  - 仅在安全边界内启用新 kernel：`S_q % 16 == 0`、`S_q % block_size == 0`、`S_k % block_size == 0`、`block_size % 16 == 0`。
  - 不满足边界时回退旧 `build_sparse_attention_qkv_fwd(..., use_contiguous_range_load=...)`。
  - 新增回退开关：`MINFERENCE_BLOCK_SPARSE_TILELANG_H1=0` 可强制旧 padded-H kernel，便于 A/B benchmark。
- `benchmarks/bench_tilelang_h1_sparse_attention.py`
  - 新增 isolated micro-benchmark，对比旧 padded-H=1 kernel 和新 query-block H=1 kernel。
- `tests/test_block_sparse_kernel.py`
  - 新增 H=1 query-block 启用/禁用策略测试。
- `examples/run_hf_minimal.py`
  - 新增 `--empty-cache-between-runs`，仅用于 64K 等长上下文重复 generate 调试，避免 HF 4D mask 重复分配/缓存碎片导致第二轮 OOM；默认关闭，不改变已有 benchmark 口径。

验证：

- `python -m py_compile minference/ops/block_sparse_kernel_npu.py minference/ops/tilelang_sparse_attention_h1.py tests/test_block_sparse_kernel.py benchmarks/bench_tilelang_h1_sparse_attention.py`：通过。
- `PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python -m pytest tests/test_block_sparse_kernel.py -q`：35 passed，1 warning。
- `PYTHONPATH=$PWD:~/tilelang-ascend conda run -n flexhead-tl python tests/test_tilelang_sparse_attention_h1.py --case all`：通过。

isolated micro-benchmark：

| S | D | topk blocks | old padded-H=1 | new query-H=1 | speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | 64 | 1 | 1.077 ms | 1.048 ms | 1.03x |
| 512 | 64 | 2 | 1.106 ms | 1.065 ms | 1.04x |
| 512 | 64 | 4 | 1.153 ms | 1.067 ms | 1.08x |
| 1024 | 64 | 1 | 1.113 ms | 1.070 ms | 1.04x |
| 1024 | 64 | 2 | 1.236 ms | 1.106 ms | 1.12x |
| 1024 | 64 | 4 | 1.945 ms | 1.063 ms | 1.83x |
| 2048 | 64 | 1 | 1.490 ms | 1.175 ms | 1.27x |
| 2048 | 64 | 2 | 2.056 ms | 1.125 ms | 1.83x |
| 2048 | 64 | 4 | 3.847 ms | 1.084 ms | 3.55x |
| 4096 | 128 | 16 | 29.919 ms | 2.224 ms | 13.46x |

所有 isolated case 的 `old_vs_new_diff` 都是 `0.0000e+00`。

wrapper benchmark（`bench_tilelang_long_context.py`，`MINFERENCE_BLOCK_SPARSE_MASK_MAX_SEQ=0`，`D=128`，`topk_blocks=16`）：

| S | kernel | mean |
|---:|---|---:|
| 4096 | new H=1 query-block | 3.37 ms |
| 4096 | old padded-H (`MINFERENCE_BLOCK_SPARSE_TILELANG_H1=0`) | 31.43 ms |
| 8192 | new H=1 query-block | 5.67 ms |
| 16384 | new H=1 query-block | 10.46 ms |
| 32768 | new H=1 query-block | 20.71 ms |
| 65536 | new H=1 query-block | 41.08 ms |

32K/64K wrapper 结果已写入：

```text
benchmarks/results/block_sparse_h1_32k_64k.json
```

4K forced TileLang HF probe（Phi-3-mini-128k-instruct，block_sparse probe config，`MINFERENCE_BLOCK_SPARSE_MASK_MAX_SEQ=0`，`--num-runs 3`）：

- run1：15.76s（含 JIT）
- run2：0.41s
- run3：0.41s
- 对比上一轮 forced TileLang 4K 稳态 `~1.62s`，端到端 forced path-B 约 3.95x 改善。

短/中上下文 HF probe（同一 block_sparse probe config，除特别说明外 `MINFERENCE_BLOCK_SPARSE_MASK_MAX_SEQ=0` 强制 TileLang path-B）：

| ctx | run2/run3 steady | notes |
|---:|---:|---|
| 4K | 0.41s / 0.41s | forced TileLang；接近 path A，但仍慢于 checkpoint 中 path A `0.32s` |
| 8K | 0.91s / 0.91s | forced TileLang；慢于 checkpoint 中 path A `0.75s` |
| 16K | 2.44s / 2.45s | forced TileLang；略慢于 checkpoint 中 path A `2.24s` |
| 32K | 7.33s run2 | 默认 path-B；接近 checkpoint 中 dense baseline `6.67s`，但尚未超过 |
| 64K | 53.53s run2 | `device_map=auto` + `--empty-cache-between-runs`；未加该开关时第二轮因 HF 4D mask 再分配 32GiB OOM |

结论：

- H=1 query-block kernel 已经实质打掉旧 path-B 的 padded-H 主瓶颈；越接近真实 `D=128/topk=1024`，收益越明显。
- 4K forced TileLang 已从完全不可用的 `1.62s` 拉到 `0.41s`，接近短序列 path A 口径（checkpoint 中 4K path A run2 `0.32s`）。
- 8K/16K forced TileLang 仍慢于短序列 path A，因此 **暂时不要降低/取消 `MINFERENCE_BLOCK_SPARSE_MASK_MAX_SEQ=16384` 默认阈值**。
- 32K 默认 path-B 已经能跑到接近 dense baseline，但端到端还没赢；64K 端到端主要被 HF 4D causal mask、跨卡/公共开销和 dense-others 拖住，kernel wrapper 本身 64K 只有 `41.08ms`。
- 下一步不要再纠缠 H=1 kernel 主体；更应该处理 HF 端 O(S²) mask 构造/复用，以及扩大 sparse 覆盖率或减少 dense-others。

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
