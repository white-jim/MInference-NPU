# MInference 1.0 → 昇腾 NPU 迁移实施方案 v1

> 用途：把"调研结论"落成可执行的阶段计划。本文只做**步骤 + 关键点**，不展开实现细节（实现细节留到对应阶段动手时再细化）。
> 创建日期：2026-05-23
> 关联文档：`docs/MInference_1.0_implementation.md`、`docs/ascend_migration_survey.md`、`docs/context_checkpoint.md`

---

## 0. 决策摘要（v1 已拍板）

| 维度 | 选择 | 理由 |
|---|---|---|
| 迁移范围 | **三种稀疏模式 + dense fallback**（vertical_and_slash / block_sparse / stream_llm / dense） | 最小可验证集，先把主路径跑通；dilated / static / tri_shape / kvcompress / dist_ops 不在 v1 |
| 算子主路径 | **Triton-Ascend** | 与现有 Triton kernel 同语言，改造成本最低；性能可达 AscendC 80%–90% |
| `convert_vertical_slash_indexes` | **Triton-Ascend 重写** | 避免 H2D 来回；与主 kernel 同部署链路 |
| 框架宿主 | **HF transformers + torch_npu** | 与原 `patch_hf` 入口对称，硬件差异收敛在 4 个 op 上 |

**约束**：CANN ≥ 8.3.RC1、torch-npu ≥ 2.6.0.RC1、triton-ascend 与之匹配。

**关于多 GPU / 多 NPU**：MInference 算法本身**与并行方式正交**（per-head 选 pattern + 每 head 独立 sparse kernel，不依赖单卡）。原仓库已支持 HF Pipeline Parallel（无感）、vLLM Tensor Parallel（`minference_patch_vllm_tp`，需手动改 vllm/worker）、Sequence Parallel（`dist_ops/`）。

**v1 多卡支持范围**：
- ✅ **HF accelerate 自动多卡部署**（`device_map="auto"` / `infer_auto_device_map`，按 layer 切到多张 NPU 上）：**直接可用**。因为 accelerate 在 layer 边界切分，每一层的 attention 仍在单卡内完整跑，per-head 循环 / 稀疏 kernel / 索引展开都在层内闭环，accelerate 只在层间做 D2D 拷贝——对 attention 算子完全透明。这是大模型长上下文推理的常用场景，v1 明确覆盖。
- ❌ **vLLM TP / 序列并行 / 显式 DP / EP**：v1 不做，留到 v2。
- v1 设计上**不引入与多卡对立的写法**（per-head 循环保持原样、不做隐性全卡 batch 化、所有 tensor 操作 device-agnostic），保证日后接 TP/SP 时不需要推翻。

**accelerate 多卡场景的工程约束**（实现时必须守住的几条）：
1. `accelerate ≥ 0.28`，确认 `device_map="auto"` 能识别 npu:0 / npu:1。
2. 所有新写代码 device-agnostic：禁止 hard-code `.npu()` / `device='npu:0'`，必须 `tensor.device` 跟随输入。
3. 新写的 Triton-Ascend kernel 启动时 device 跟随输入 tensor，不要自行指定。
4. 长上下文场景 KV cache 体积可能远大于参数，accelerate 默认按参数切分，可能需要手动给 `max_memory` 预留 KV 余量（配置层面解决，非代码改动）。

---

## 0.5 项目组织与产出形态

**迁移后的代码独立成一棵新树** `D:\works\算法迁移\MInference-NPU\`，与上游 `MInference/` 平级、不混在一起。要求是**完整实现**：

- 拷贝原 `MInference/minference/` 中 v1 范围内要用到的全部 py 文件（patch、forward、ops、configs、utils、modules 等），即使没改动也要拷。
- 新写或改写的 Triton-Ascend kernel、`npu_fusion_attention` 适配代码放在对应位置（如 `MInference-NPU/minference/ops/`、`MInference-NPU/minference/backend_npu/`）。
- `MInference-NPU/setup.py` 单独维护，**不编译 CUDA extension**；该树能独立 `pip install -e .` 跑起来。
- 测试 / 示例脚本归到 `MInference-NPU/tests/` 与 `MInference-NPU/examples/`。
- 原 `MInference/` 保留作为参考蓝本，不动。

**目录草图（v1 完成时大致样子）**：

```
D:\works\算法迁移\
├── MInference/                  ← 原始 CUDA 版（参考用，不动）
├── MInference-NPU/              ← v1 产出
│   ├── minference/
│   │   ├── ops/                 ← Triton-Ascend kernel + 索引展开
│   │   ├── backend_npu/         ← npu_fusion_attention 等 API 的薄封装
│   │   ├── modules/             ← minference_forward.py（NPU 改造版）
│   │   ├── patch.py             ← patch_hf 的 NPU 版
│   │   ├── configs/             ← best_pattern JSON（沿用上游）
│   │   └── ...
│   ├── tests/                   ← 模块级 + 端到端测试脚本
│   ├── examples/                ← 最小调用示例 + 效果测试脚本
│   ├── docs/                    ← 各模块的操作说明（README 风格）
│   └── setup.py
└── docs/                        ← 本系列方案/调研/检查点文档
```

---

## 0.6 优先复用调研成果（开发前必查）

调研文档 `docs/ascend_migration_survey.md` 已经盘清楚：**MInference 1.0 的 4 个核心组件每一个都有可复用的昇腾资源**。每个 milestone 动手前，先到对应行查"已经有什么"，能不写就不写。

| MInference 组件 | 昇腾上可复用的现成成果 | 落到本方案的阶段 | 工作量评级 |
|---|---|---|---|
| Dense FA（fallback / decode） | `torch_npu.npu_fusion_attention` / `npu_prompt_flash_attention` / `npu_incre_flash_attention` / `npu_fused_infer_attention_score` | **M1** | 零 |
| Streaming（A-shape） | `npu_fusion_attention` 的 `sparse_mode=4` band-causal + prefix；一行替换 | **M2** | 极低 |
| Block-Sparse FA | `npu_fusion_attention` + `atten_mask`（兜底）；或 TileLang `examples/blocksparse_attention/`（参考骨架） | **M3** | 低 |
| Vertical-Slash FA（主路径） | **TileLang-Ascend `examples/sparse_flash_attention/`**（DSA 同构骨架）+ **cann-recipes-infer `npu_sparse_flash_attention.py`**（AscendC 参考）+ **FastAttention 论文 (arXiv:2410.16663)** 的 two-level tiling | **M4-b** | 中-高 |
| `convert_vertical_slash_indexes` | 无 1-to-1；但有相邻范式：vllm-ascend RFC #5507 `hamming_dist_top_k`、RFC #807 `advance_step` AscendC kernel、Lightning Indexer | **M4-a** | 中 |
| Per-head 调度 + best_pattern JSON | 纯 Python，**直接复用上游** | M1 拷贝即可 | 零 |
| 工具链辅助 | `msit precision_compare`（精度比对）、`msprof`（性能 profile）、AOE（自动调优） | **M5** | 零 |
| HF 层 flash_attn 透明转发参考 | `kernels-ext-npu/flash-attn2`（HuggingFace） | **M1** patch 改造时参照其风格 | 零 |

**使用规则**：
1. 每个 milestone 开工前，先打开 `docs/ascend_migration_survey.md` §3 对应小节，确认是否有官方 API / 公开仓库能直接用或当蓝本。
2. **优先级**：官方 torch_npu API > TileLang-Ascend / cann-recipes-infer 蓝本 > Triton-Ascend 自写 > AscendC 手写。
3. 调研文档里 §5 的"开源仓库速查表" 与 §6 的"学术资料速查"是兜底参考，遇阻时拉出来看。
4. 复用不是照抄——蓝本仓库的接口、dtype、layout 与 MInference 不一定吻合，需要做薄适配层；但**算法骨架、tile 策略、跨核同步范式**这些大头能省一半工。

---

## 1. 总体路线图

```
M0 环境与依赖           ─┐
M1 上层 Python 链路通    │ ─→ 不依赖自写 kernel，先用 torch_npu API 把端到端跑起来
M2 Streaming kernel      │
M3 Block-Sparse kernel   │ ─→ 由易到难，三种稀疏依次迁
M4 Vertical-Slash kernel │     （含 convert_vertical_slash_indexes 重写）
M5 端到端联调 + 精度/性能 ─┘
```

**节奏建议**：M0–M1 一周内打通；M2–M3 各两到三天（kernel 相对简单）；M4 是大头（占整体一半工作量）；M5 持续到验收。

---

## 2. M0 — 环境与依赖准备

**目标**：在目标昇腾机器上跑通一个最小 Triton-Ascend kernel，确认工具链可用。

**主要任务**
- 装 CANN ≥ 8.3.RC1、`torch_npu` 对应版本、`triton-ascend`。
- 创建 `MInference-NPU/` 目录，搭出空骨架（`minference/`、`tests/`、`examples/`、`docs/`、`setup.py`），先把上游 v1 范围内的 py 文件原样拷过来作为起点。
- 起个最小 vector add 的 Triton kernel，在 NPU 上跑通，确认编译 / 调度链路无误。
- 起个最小 `torch_npu.npu_fusion_attention` 调用，确认 dense FA API 可用。

**测试与说明**
- `MInference-NPU/tests/test_env.py`：跑 vector add + 最小 `npu_fusion_attention`，输出 PASS/FAIL。
- `MInference-NPU/docs/SETUP.md`：环境安装步骤、版本要求、`test_env.py` 怎么跑。

**关键点**
- **目录**：源码本体仍在 `D:\works\算法迁移\MInference\`，所有改动都在这棵树里做；昇腾相关新增文件用前缀 `_ascend` 或独立子目录区分。
- **不依赖** `flash-attn` / `sgl_kernel` / `vllm_flash_attn` —— 这些在 NPU 上不可用，所有 import 必须有 try-except 守护。
- 把环境信息（CANN 版本、torch_npu 版本、triton-ascend commit）记到 `docs/context_checkpoint.md` 的"环境信息"段。

**验收**：能在 NPU 上跑通"vector add Triton kernel"与"`npu_fusion_attention` dense 调用"，输出与 CPU 参考一致。

---

## 3. M1 — 上层 Python 链路打通（不动 kernel）

**目标**：在 NPU 上能把一个真实 LLM 端到端推完一次 prefill，但 attention 路径**全部走 dense**（即每条分支都退化为 `npu_fusion_attention`）。这样把所有"非算子"问题先消干净。

**主要任务**
- `minference/patch.py:patch_hf` 改造：把里面所有 `flash_attn_func` / `flash_attn_varlen_func` 替换成 `npu_fusion_attention` 等 NPU API（参考 `kernels-ext-npu/flash-attn2` 的转发风格）。
- `minference/modules/minference_forward.py:minference_forward()` 改造：
  - `q_len == 1` 分支 → `npu_incre_flash_attention` 或 `npu_fused_infer_attention_score`。
  - `layer_idx < starting_layer` 分支 → `npu_prompt_flash_attention` / `npu_fusion_attention`。
  - prefill 主路径**先让三种稀疏 kernel 都临时退化为 dense**（不动 best_pattern 加载、不动 per-head 循环、不动在线估计代码）。
- 选定基线模型：Qwen2.5-7B 或 Llama-3-8B，中等长度（32k）先跑通；千万别一上来就上 1M。
- `RotaryEmbeddingESM` 等 RoPE 改造：检查是否依赖 CUDA-only 算子（如 `torch.compile` 后端、`flash_attn` 的 `apply_rotary_emb`），需要时换 `torch_npu.npu_apply_rotary_pos_emb` 或 pure PyTorch。

**关键点**
- **保留 per-head 循环骨架不动**，只把循环体里的"调对应 kernel"暂时统一指向 dense。这样 M2–M4 每加一种稀疏 kernel 就替换一个分支。
- 离线 `configs/*.json` 直接复用，无需重新搜（v1 用现成的 Llama-3.1-8B / Qwen2.5-7B 的 best_pattern）。
- `MInference-NPU/setup.py` **不含** `CUDAExtension("minference.cuda", ...)`；所有 `from ..cuda import ...` 的 import 替换为 NPU 版索引展开的 import（M4 之前用 host fallback 占位）。
- 注意 **dtype**：NPU 上 `npu_fusion_attention` 对 fp16 / bf16 支持有别，要和模型 dtype 对齐；fp32 路径可能不通。

**测试与说明**
- `MInference-NPU/tests/test_dense_forward.py`：把所有 attn 分支强制 dense，跑一次完整 forward，断言输出与 HF 原生 `eager` attention 差异 < 1e-2。
- `MInference-NPU/examples/run_hf_minimal.py`：最小 HF 调用示例（32k 长度 prompt + greedy 解码 ~10 token）。
- `MInference-NPU/docs/M1_dense_pipeline.md`：怎么装 + 怎么跑 + 预期输出。

**验收**：端到端能跑出和原版 MInference（CUDA）**结构相同的输出**（差异容忍稍大，因为底层 attention 实现不同）；perplexity 在长 prompt 上不爆。

---

## 4. M2 — Streaming Kernel 迁移（最易，先热身）

**目标**：把 `stream_llm` 分支从 dense 替换为真正的 A-shape 稀疏调用。

**主要任务**
- 把 `streaming_forward`（`ops/streaming_kernel.py`）的接口保留，内部实现**用 `npu_fusion_attention(sparse_mode=4)` + sink 拼接**代替原 Triton kernel。
- 验证 sink + sliding window 的语义与原版等价（n_init=128, n_local=3968 这种默认值）。

**关键点**
- 这是调研里明确说"一行替换"的部分，**不要为了"完整 Triton-Ascend 迁移"而硬把 streaming kernel 翻一遍**——浪费工程量。
- sink 段若 `npu_fusion_attention` 的 `prefix` / `pre_tockens` 参数表达不了，退化为"两次 attention + 合并"的 Python 层实现也能接受。

**测试与说明**
- `MInference-NPU/tests/test_streaming_kernel.py`：随机 q/k/v + 多组 (n_init, n_local) 配置，输出与"PyTorch 参考实现（手写 mask + softmax）"逐元素对比，差异 < 1e-2。
- `MInference-NPU/docs/M2_streaming.md`：streaming 路径用法 + 测试跑法。

**验收**：跑一条 best_pattern 把所有 head 强制设成 `stream_llm` 的 case，输出与"该 pattern 在 CUDA 上的输出"差异 < 1e-2（NPU vs GPU 数值不会完全一致）。

---

## 5. M3 — Block-Sparse Kernel 迁移

**目标**：把 `block_sparse` 分支从 dense 替换为真正的 block-sparse 调用。

**主要任务**
- `_build_block_index`（`ops/block_sparse_flash_attention.py:151`）是纯 PyTorch，**直接复用**，无须改动。
- 用 **Triton-Ascend 翻写** `_triton_block_sparse_attn_fwd_kernel`（单 K-block 循环、FA online softmax，结构简单）；或先用 `npu_fusion_attention` + 动态构造 `atten_mask` 做兜底实现。
- 接口对齐：`block_sparse_attention(q, k, v, topk)` 不变。

**关键点**
- 这是 **Triton 翻 Triton-Ascend 的第一块练手代码**——结构简单、没有两段 softmax，先把"Triton-Ascend 的内存访问、tl.dot、tl.load/store"这一套熟悉清楚，再上 M4 的硬骨头。
- **UB / 片上存储约束要算清**：`BLOCK_M × BLOCK_DMODEL × 2byte`（fp16）加上 K/V 缓冲 + 中间 acc，估算是否塞得下；不行就把 BLOCK_M / BLOCK_N 调小。
- 关注 NPU 的"Cube / Vector 分离"：A2/A3 上 `tl.dot` 走 Cube，softmax 走 Vector，编译器是否自动交错要 profile 一下。

**测试与说明**
- `MInference-NPU/tests/test_block_sparse_kernel.py`：固定 topk + 随机 q/k/v，断言输出与"PyTorch 参考稀疏 attention（按 block_index gather 后做 dense softmax）"差异 < 1e-2。
- `MInference-NPU/docs/M3_block_sparse.md`：kernel 接口、tile 大小约束、测试跑法。

**验收**：单 head + 单层 block_sparse 的输出与 CUDA 版差异 < 1e-2。

---

## 6. M4 — Vertical-Slash Kernel 迁移（**核心难点**）

这一块是整个 v1 工作量的一半左右，**拆成两步**做：

### M4-a — `convert_vertical_slash_indexes` 重写

**目标**：把 CUDA 双指针索引展开 kernel 用 Triton-Ascend 重写，保持输入输出形状不变，作为 NPU 端 op。

**主要任务**
- 接口对齐（输入 `seqlens / v_idx / s_idx`，输出 `block_count / block_offset / column_count / column_index`）。
- 用 Triton-Ascend 实现"slash 合并 + vertical 去重"双指针逻辑。
- 替换 `pit_sparse_flash_attention_v2.py:195` wrapper 里的 `from ..cuda import convert_vertical_slash_indexes` import。

**关键点**
- **dynamic shape 是难点**：每步 v_idx / s_idx 形状可变，Triton-Ascend 下的 host tiling 是否需要 re-jit 要确认；最坏情况下针对 `NNZ_V / NNZ_S` 取一组离散尺寸预编译。
- **控制流密集**：原 CUDA kernel 大量分支（合并 range / 去重判断），Triton 表达上要把"循环 + 条件写"转成"mask 化"才能跑得快；写不好就退化为串行。
- **第一版可以接受不优**：只要正确性 OK，性能瓶颈在 M5 profile 后再回头优化。
- 备选保险路径：若 Triton-Ascend 实现遇到无法解决的语义障碍，**临时退回 PyTorch CPU 实现**（每步多一次 H2D，但能让 M4-b 继续推进），保留该分支以便切换。

**测试与说明**
- `MInference-NPU/tests/test_index_expand.py`：构造若干 (seqlens, v_idx, s_idx) 输入，与原 CUDA 输出 / 或 Python 黄金参考实现做逐元素 exact match。
- `MInference-NPU/docs/M4a_index_expand.md`：算法说明（双指针 + slash 合并 + vertical 去重）+ 测试跑法。

**验收**：随机 v_idx / s_idx 输入，输出与 CUDA 版逐元素一致（这一步是确定性算法，应该能 exact match）。

### M4-b — `_triton_mixed_sparse_attn_fwd_kernel` 重写

**目标**：把两段 online softmax 的 vertical-slash kernel 用 Triton-Ascend 翻写。

**主要任务**
- 接口对齐：`vertical_slash_sparse_attention(q, k, v, v_idx, s_idx)` 不变。
- 翻写 `pit_sparse_flash_attention_v2.py:48` 的 `_triton_mixed_sparse_attn_fwd_kernel`：
  1. 段 1：按 `block_offset` 扫连续 K 块。
  2. 段 2：按 `column_index` gather vertical 列。
  3. 两段共享 `m_i / l_i` 维护 online softmax。
- **优先参考 TileLang-Ascend `examples/sparse_flash_attention/`** 与 **cann-recipes-infer `npu_sparse_flash_attention.py`** 的内存布局与 tiling 思路；不必照搬，但要对照他们怎么处理"两段访问"。

**关键点**
- **`exp2 + log2(e)` 替代 `exp`**：CUDA Triton 上是 CSE/LICM 的工程 trick，Triton-Ascend 上**先验证 `tl.exp` 行为**再决定要不要保留这个变换；不必盲目沿用。
- **gather 访问**（段 2 的 `tl.load(K[..., cols])`）在 NPU 上要确认 Triton-Ascend 是否生成高效 MTE 指令；若不行考虑改为"先在 host 把 vertical 列拼成连续 buffer 再喂 kernel"。
- **padding 处理**：原版用 F.pad 把 context_size 补齐到 `block_size_M` 的倍数；这一步 Python 侧逻辑不变。
- **head_dim 必须是 2 的幂**：原版 pad 到 2 的幂，沿用这一约束。
- **causal mask**：段 1 需要因果裁剪（`cols <= offs_m`），与 CUDA 版本一致；段 2 vertical 列不带因果（因为索引展开已经做过）。
- **per-head 调度**：保持原本的 Python 外循环（每个 head 独立调 kernel），暂不做 head 维度 batch 化。

**测试与说明**
- `MInference-NPU/tests/test_vertical_slash_kernel.py`：随机 q/k/v + 随机 v_idx/s_idx，断言输出与"PyTorch 黄金参考（按索引拼 K/V 后跑 dense softmax）"差异 < 1e-2。
- `MInference-NPU/tests/test_vertical_slash_e2e.py`：固定 best_pattern 跑单层完整 forward，与原版 CUDA 单层输出对比。
- `MInference-NPU/docs/M4b_vertical_slash.md`：两段 online softmax 算法说明 + 接口 + 测试跑法。

**验收**：固定 best_pattern（v_size=1000, s_size=6096）下，单层输出与 CUDA 版差异 < 1e-2；32k 长度下能跑完一次完整 prefill。

---

## 7. M5 — 端到端联调 + 精度 / 性能 / 效果验证

**目标**：把 M2/M3/M4 三个 kernel 同时接回去（不再退化为 dense），跑完整 best_pattern 配置下的端到端推理，做精度对比、初步性能 profile，并出一个**基本的效果测试**。

**主要任务**
- 联调：把 `gather_last_q_vertical_slash_topk_v4` 里的三个分支全部指向新 NPU 实现。
- **精度比对**：
  - 用 `msit precision_compare` 对比 NPU vs CUDA 的逐层 hidden states。
  - 跑 needle-in-a-haystack 或 RULER 子集，定性看准确率有无塌方。
- **性能 profile**：
  - `msprof` 看 Cube / Vector / MTE 利用率。
  - 拆 latency：QKV proj / 索引展开 / 段 1 扫块 / 段 2 gather / output。
  - 与原 CUDA + Triton 在等价 GPU 上的 latency 做横向对比。
- **回归测试**：选 2–3 个长度（32k / 128k / 1M 视显存而定）跑通；不在 v1 范围内的 attn_type（dilated/static/...）要么禁用要么保留 fallback 路径。
- **基本效果测试**（必交付，不必专业）：
  - 一个长 prompt 的"针在干草堆"（needle-in-a-haystack）小用例：把一句关键信息埋进 32k / 128k token 文本，看模型能否检索回答。
  - 几条短 QA 任务做 sanity check，保证 NPU 版回答连贯、没有崩坏成乱码。
  - 同模型同输入跑原版 CUDA + 新版 NPU，输出文本做人眼对照。

**测试与说明**
- `MInference-NPU/tests/test_e2e_smoke.py`：跑通端到端一次 prefill + 解码 ~20 token，断言不挂、不 NaN。
- `MInference-NPU/tests/test_e2e_multi_npu.py`：用 `accelerate` + `device_map="auto"` 把模型自动切到 2 张 NPU 上跑一次完整 prefill（smoke 级别，确认不挂、device-agnostic 约束没破）。**前提是有 ≥ 2 张 NPU 可用**，单卡环境跳过。
- `MInference-NPU/examples/effect_test_needle.py`：针在干草堆基本效果测试脚本，可指定 prompt 长度。
- `MInference-NPU/examples/effect_test_qa.py`：几条简单 QA，跑完打印 NPU 输出 vs CUDA 参考输出对照。
- `MInference-NPU/docs/M5_e2e_and_eval.md`：怎么跑联调、怎么跑效果测试、怎么读精度 / 性能报告，**含 accelerate 多卡跑法说明**。

**关键点**
- **NPU vs GPU 输出不要求 bit-exact**。判定标准：max abs diff、相对 diff、下游任务 score。
- 若 vertical-slash kernel 性能没达预期，按优先级处理：
  1. 先确认 `convert_vertical_slash_indexes` 是不是瓶颈（若是，启用 host CPU fallback 或并行化 host 实现）。
  2. 再看段 2 gather 的访存效率。
  3. 最后考虑 BLOCK_M / BLOCK_N tile 大小调参 / AOE 自动调优。

**验收**：
- 长上下文 prefill 能稳定跑通（不挂死、不 OOM）。
- 输出 perplexity 与 CUDA 版差异在论文报告的精度范围内。
- 给出一份"NPU vs GPU latency 对照表"作为 v1 交付物。

---

## 8. 关键技术点 / 全局风险（贯穿整个迁移）

1. **Triton-Ascend 与 CUDA Triton 的语义差异**：编译器后端不同，部分 builtin（如 `tl.exp` / `tl.dot` 的精度路径 / `tl.atomic_*`）行为可能有别；遇到不对劲的地方先查 triton-ascend 文档，再考虑变通。
2. **dynamic shape**：MInference 的稀疏索引每步变化，最坏情况会触发反复 re-tiling。M4-a 的索引展开是重灾区，必要时按 NNZ 离散尺寸预编译。
3. **UB / 片上存储约束**：所有 kernel 的 tile 配置必须显式估算 UB 占用，不要照搬 CUDA Triton 上 work 的常量（Hopper / Ampere 的 shared mem 配额与 NPU UB 容量量级不同）。
4. **Cube / Vector 分离（A2/A3）**：跨核同步需要显式 flag；Triton-Ascend 大部分是自动处理，但 profile 时若发现 Cube/Vector 串行执行，要考虑 hint 编译器交错。
5. **dtype 一致性**：bf16 是 NPU 推理推荐 dtype；fp16 走得通但要测溢出；fp32 走不通的算子要切到 bf16 上。
6. **`flash-attn` 系列 import**：原代码大量 import flash_attn / sgl_kernel / vllm_flash_attn，**必须**用 try-except 守护并提供 NPU fallback，否则 import 阶段就挂。
7. **CUDA extension 构建**：v1 暂不需要 `minference.cuda`；构建脚本要能在 NPU 环境下跳过该扩展。后续若做 CUDA / NPU 并存版本再处理。
8. **best_pattern 兼容性**：现成 JSON 是按特定模型搜的，不要换模型；若必须换，用 `is_search=True` 在 GPU 上重新搜后再迁过来（不要在 NPU 上从零搜，搜索本身就慢）。
9. **device-agnostic**（accelerate 多卡支持的隐含要求）：所有 tensor 创建、kernel 启动、临时 buffer 都必须跟随输入 tensor 的 device，不写死 `npu:0`。原版 MInference 这点不严格（依赖默认 device），迁移时统一收紧，否则 accelerate 切层时同一层的输入 / 中间变量 device 不一致会直接挂。

---

## 9. 不在 v1 范围（明确排除）

下列项目**不做**，避免 scope creep：

- `dilated1` / `dilated2` / `static` / `tri_shape` / `vs_only` 等消融用 attn_type。
- KV 压缩（`snapkv` / `pyramidkv` / `quest` / `kivi` / `retr_attn` / `leank` / `streamingllm`）。
- 分布式（`dist_ops/` 下的 Ring / Striped / Zigzag）。
- vLLM-Ascend 集成、MindIE 集成、`patch.py:minference_patch_vllm*` 路径。
- FA3 / Hopper-style 的 `pit_sparse_flash_attention_v3.py`（约 1500 行，v1 不动）。
- 离线 best_pattern 搜索（`search_pattern` / `_v2`）在 NPU 上的复现 —— 沿用 GPU 已搜结果。

这些在 v1 跑通后视需求再起 v2 计划。

---

## 10. 交付物清单

v1 完成时应该有：

1. **完整代码树** `D:\works\算法迁移\MInference-NPU\`：可独立 `pip install -e .` 跑通，包含
   - v1 范围内的全部 Python 源文件（patch / forward / ops / configs / utils / modules / backend_npu）
   - 三个新写的 Triton-Ascend kernel：`convert_vertical_slash_indexes` / vertical-slash 主 kernel / block-sparse kernel
   - `MInference-NPU/setup.py`（无 CUDAExtension）
2. **模块级测试** `MInference-NPU/tests/`：
   - `test_env.py` / `test_dense_forward.py` / `test_streaming_kernel.py`
   - `test_block_sparse_kernel.py` / `test_index_expand.py`
   - `test_vertical_slash_kernel.py` / `test_vertical_slash_e2e.py` / `test_e2e_smoke.py`
3. **示例与效果测试** `MInference-NPU/examples/`：
   - `run_hf_minimal.py`、`effect_test_needle.py`、`effect_test_qa.py`
4. **模块文档** `MInference-NPU/docs/`：每个 milestone 一个 `Mx_*.md`，说明该模块用法、测试跑法、预期结果。
5. **流程与报告**（写在外层 `docs/`，便于与方案文档对照）：
   - `docs/migration_v1_notes.md`：迁移过程踩坑记录、与本计划的偏差。
   - `docs/migration_v1_report.md`：精度比对表 + latency 对照表 + 基本效果测试结果。

---

## 11. 下一步动作

按时间线，**立刻可以开始的是 M0**：

1. 确认目标 NPU 机器、CANN / torch_npu / triton-ascend 的版本。
2. 把 `MInference/` 目录 `git init` 起来，建一个 `ascend-v1` 分支。
3. 跑最小 Triton-Ascend kernel 验证工具链。

如果 M0 阶段遇到环境 / 版本 / 选型上需要重新拍板的问题，回到本计划相应章节修订即可。
