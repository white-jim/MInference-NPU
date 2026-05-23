# 昇腾 NPU 迁移调研 — Ascend C 算子开发现状 & MInference 1.0 可复用成果盘点

> 用途：在动手设计 MInference 1.0 → 昇腾 NPU 迁移方案前，先把"目标平台的开发方式"和"别人已经做过哪些事"摸清楚，避免重复造轮子。
> 创建日期：2026-05-23
> 关联文档：`docs/context_checkpoint.md`、`docs/MInference_1.0_implementation.md`
> 范围：本文档**只做调研，不做方案设计**。

---

## 0. TL;DR

1. **昇腾 NPU 上写算子有 3 条主流路径**，从底到顶分别是：
   - **Ascend C**（C++ 风格，类似 CUDA，手写 tiling / 队列同步，性能上限最高、开发量也最大）
   - **Triton-Ascend**（OpenAI Triton 的 NPU 后端，毕昇编译器把 Triton IR → AscendNPU IR；与 MInference 现有 Triton kernel **同语言**，是改造成本最低的路径）
   - **TileLang-Ascend**（Pythonic DSL，2025-09 开源，已有 FlashAttention / **SparseFlashAttention** / LightningIndexer 的参考实现）
2. **整套 MInference 1.0 在昇腾上的公开移植目前不存在**（未在 Gitee / GitHub / GitCode 搜到）。但是构成 MInference 1.0 的 4 个关键算子，**每一个都已有对应的、可参考的 NPU 实现或框架支持**：
   | MInference 1.0 组件 | 昇腾上可直接复用的成果 |
   |---|---|
   | `flash_attn_func`（dense fallback / decoding） | `torch_npu.npu_fusion_attention`、`npu_prompt_flash_attention`、`npu_incre_flash_attention`、CANN `npu_fused_infer_attention_score` |
   | Block-Sparse FlashAttention | `npu_fusion_attention` 的 `sparse_mode` + `atten_mask`（mode 0/2/3/4/6/7）；TileLang-Ascend `examples/blocksparse_attention/` |
   | Vertical-Slash / 通用稀疏 FA | **TileLang-Ascend `examples/sparse_flash_attention/`（含 SparseFlashAttention 与 LightningIndexer，AscendC PTO 实现对标 cann-recipes-infer 的 `npu_sparse_flash_attention.py`）** —— 与 MInference 思路高度相近 |
   | `convert_vertical_slash_indexes` 索引展开 CUDA kernel | 暂无 1-to-1 对应；可参考 vllm-ascend RFC #5507 中的 `hamming_dist_top_k` 等自定义 NPU kernel 范式，或用 Triton-Ascend / TileLang-Ascend 直接重写 |
3. **学术界已经发表过 FA2 → Ascend 的完整迁移工作**（**FastAttention** arXiv:2410.16663），披露的 two-level tiling / tiling-mask / tiling-AllReduce 策略可直接借鉴；以及华为内部的 **AMLA**（arXiv:2509.25224）在 Ascend 910 上实现了 86.8% 峰值算力的 FlashMLA，已并入 CANN。
4. **整体推理框架层**，vLLM-Ascend、MindIE、MindSpeed-LLM 都在持续完善长上下文 prefill 路径，特别是 vLLM-Ascend 已有 KVComp 稀疏注意力的 RFC（Issue #5507），是接入"per-head 稀疏注意力"的天然宿主。

---

## 1. 昇腾 NPU 平台基础（迁移者必备）

### 1.1 硬件抽象

- **AI Core** 是昇腾 NPU 的计算核心，一个 NPU 内有多颗 AI Core。Ascend C 程序通过内置变量 `block_idx` 在多核之间做数据切分（SPMD 模型）。
- AI Core 内部三类执行单元：
  - **Cube Unit**：矩阵乘，每次 fp16 16×16 × 16×16；大矩阵需切块。
  - **Vector Unit**：逐元素、归约等，受限于 `Unified Buffer (UB)` 容量与对齐（32B）。
  - **Scalar Unit**：地址计算、循环控制、分支。
- **关键架构演进**：A2/A3 上 **Cube 与 Vector 已分离成独立物理核**，跨核通信走 L2/GM，需要显式同步（如 TileLang 的 `T.set_cross_flag` / `T.wait_cross_flag`）。这与 CUDA 上"同一个 SM 内 tensor core + CUDA core"的合并模型不同，是迁移时容易踩的坑。
- **片上存储多级**：GM (HBM) → L1 → L0A/L0B/L0C (Cube 专用 scratchpad) → UB (Vector 专用)。容量随型号差异较大（**具体数值见 hiascend.com 的型号架构文档**；公开资料中 UB 通常在百 KB 到 MB 量级）。**所有片上存储显式管理，搬运由 MTE (Memory Transfer Engine) 子系统调度**，CUDA / Triton 程序员的"shared memory 自动调度"直觉在这里不成立。

### 1.2 Ascend C 编程范式

源自昇腾社区与官方文档 / 教程，是写 NPU 算子最底层、最可控的路径。

**三段流水任务**：

```
CopyIn  ──> Compute ──> CopyOut
 (GM→UB)     (Vector / Cube on UB / L0)    (UB→GM)
```

- 任务间通过 **Queue（VECIN / VECOUT 等 QuePosition）** 同步：`EnQue / DeQue`。
- 内存由 **Pipe** 统一管理：`InitBuffer` 给 Queue 分配池子；运行时 `AllocTensor` / `FreeTensor`。
- 中间临时变量用 **TBuf**（只能计算、不入出队列）。
- 多 tile 切分后，编译器自动产生跨 tile 的"软件流水"，达到 CopyIn[i+1] 与 Compute[i] 并行的效果（双 buffer，Ping-Pong）。

**API 分层**：
- **基础 API**：直接对应硬件能力（`Add`, `Mul`, `Mmad`, `DataCopy`, `Duplicate` …）。
- **高阶 API**：组合基础 API 完成常用算法（`Softmax`, `LayerNorm`, `MatMul` 模板等）。

**孪生调试（重要差异点）**：
- 同一份 kernel 代码既能编进 NPU 又能编进 CPU 模拟（GCC + `ASCENDC_CPU_DEBUG` 宏区分头文件路径）。
- CPU 模式可以用 `printf`、GDB 单步调试，逻辑跑通后再上 NPU 调性能。这一能力是昇腾对比 CUDA 的工程优势——**没有 NPU 卡也能开发**。

**自定义算子的完整链条**（参考昇腾社区教程 + MindSpore 集成文档）：

1. 算子分析（数学表达式、输入输出 shape/dtype）
2. 核函数实现（`__global__` 风格的 `KernelEntry`）+ 类 `KernelXxx` 封装 Init/Process/CopyIn/Compute/CopyOut
3. **Host 侧 Tiling 函数**（CANN 特色：tiling 计算运行在 CPU 上，决定 BlockDim 与 TileSize，结果作为参数传入 kernel）
4. CPU 域验证（孪生调试）→ NPU 域验证（`aclrtlaunch_<kernel_name>.h`）
5. UT / ST 测试（`msopst` 一键打包成 .om 跑真实硬件）
6. 集成到框架：PyTorch（`torch_npu` 注册 custom op） / MindSpore（`Custom` 原语 AOT 类型）

### 1.3 工具链一览

| 工具 | 用途 |
|---|---|
| **CANN Toolkit** | 编译器 (毕昇 `ascendc++`)、运行时 (ACL)、Profile / GE / FE 等 |
| **NNAL / ATB**（Ascend Transformer Boost） | 高阶推理算子库；安装在 `${CANN}/nnal/atb/`，类似 NV 的 CUTLASS + FlashInfer |
| **msFmkTransplt** | PyTorch / TF 模型迁移分析工具（算子支持度、不兼容 API） |
| **AOE** | 离线自动调优（Tiling / Knobs 搜索） |
| **msit** | 统一推理调试调优工具链（精度比对、性能 profile） |
| **Triton-Ascend** | OpenAI Triton 的 NPU 后端（毕昇 → AscendNPU IR） |
| **TileLang-Ascend** | Pythonic DSL，自动 Pipeline / 跨核同步 / Buffer reuse |
| **MindIE** | 华为官方 LLM 推理引擎（闭源 SDK，对标 TensorRT-LLM） |
| **MindSpeed-LLM** | 训练框架（Megatron-LM 的 Ascend fork，开源） |

### 1.4 别人的经验与坑（社区精华）

汇总自昇腾社区技术文章、CSDN 鲲鹏昇腾社区、知乎专栏：

1. **片上内存是第一约束**。所需 UB = (输入缓冲 + 输出缓冲 + 中间变量) × 数据宽度 × Tile × **双缓冲系数**。超限会静默挂死。
2. **Tile 切分思想是核心思维转变**。开发者反馈："真正的挑战从来不是语法，而是如何把数学公式高效映射到昇腾的物理世界"。
3. **复杂算子数据流冗长**。一个 Conv2D 涉及 GM → L1/UB → [Im2Col+Padding] → L1 → L0A/L0B → Cube → L0C → UB → [Bias+Act] → GM，每一段都要显式编排。
4. **动态 shape 是历史痛点**。Host 侧 Tiling 函数对每种 shape 都要算一次；如果 shape 频繁变化，会陷入 "重新 tiling → 重新编译"。MInference 的稀疏索引每步变化，这一点必须考虑。
5. **Cube/Vector 资源竞争**：A2/A3 拆开成独立核后，反而要主动制造"既算 GEMM 又做 softmax"的指令交错，让两类核并行起来。
6. **建议先 CPU 域跑逻辑、再上 NPU 调性能**，孪生调试能省至少一半时间。
7. **优先用高阶 API 与官方算子库**：能用 ATB 就别手写 FA，能用 `npu_fusion_attention` 就别从头造稀疏 FA。
8. **Triton-Ascend 的实践数据**：经过优化后 Triton 算子能达到手工 Ascend C 80%–90% 的性能（来源：社区开发者博客）。对于"先跑通再调优"的工程节奏极其友好。

---

## 2. 用 Ascend C 开发算子的"标准流程"（清单化）

把上面的散点收敛成一份可执行的 checklist，迁移时反复对照：

| 阶段 | 关键产出 | 工具 / 文档锚点 |
|---|---|---|
| 0. 算子分析 | 数学表达、I/O 形状/类型、与 CUDA 版的差异点 | 参考 NVIDIA 实现 + 论文 |
| 1. 选编程语言 | Ascend C / Triton-Ascend / TileLang-Ascend / 用现成 ATB API | 三者性能-工程量权衡 |
| 2. 核函数骨架 | `Init / CopyIn / Compute / CopyOut / Process`，Queue / Pipe / TBuf 完整 | 官方"3 天上手 Ascend C 编程" |
| 3. Host Tiling | `*_tiling.cpp`，决定 BlockDim / TileSize / 中间 buffer 字节数 | Tiling 函数最佳实践 |
| 4. 编译注册 | OpenJSON `op_proto.json` + CMakeLists + `register_op_check` | 自定义算子工程模板 |
| 5. CPU 孪生调试 | `ASCENDC_CPU_DEBUG=1`，GDB 单步、`printf` | 孪生调试章节 |
| 6. NPU 上跑通 | `aclrtlaunch_<name>` host 启动；对照 CUDA / numpy 参考 | ACL 接口 |
| 7. 精度比对 | `msit` `precision_compare`，dump 中间 tensor | msit |
| 8. 性能分析 | `msprof` 看 Cube / Vector / MTE 利用率、Bank 冲突、stall | msprof |
| 9. 调优 | 双 buffer、Cube↔Vector 交错、Tile 改大改小、L1 复用 | "Ascend C 高级算子优化" |
| 10. 集成到 PyTorch | 通过 torch_npu 的自定义算子注册或 `OpAdapter` | torch_npu 文档 |
| 11. UT + ST | UT 100% 分支覆盖，ST 真实硬件单算子测试 | `msopst` |

如果选 **Triton-Ascend** 路径，0–6 可大幅简化（Python 一份代码 + `@triton.jit`），但 7–9 仍然要做。
如果选 **TileLang-Ascend** 路径，自动 Pipeline / 自动跨核 flag / 自动 Buffer Reuse 都能省掉，性能上限约为 AscendC 的 90% 左右（仓库 SparseMLA benchmark 实测）。

---

## 3. MInference 1.0 相关算子在昇腾上的现成成果

下面按 MInference 1.0 的 4 个核心组件，盘点已有的可复用资源。

### 3.1 Dense FlashAttention（fallback / decoding 路径）

**现状：完全有现成的，无需自己写。**

| API / 项目 | 形态 | 适用场景 |
|---|---|---|
| `torch_npu.npu_fusion_attention` | torch_npu PyTorch 算子 | FA2 等价品；训练 + 推理；支持 `sparse_mode` 0–7 与 `atten_mask` |
| `torch_npu.npu_prompt_flash_attention` | torch_npu 算子 | **Prefill 专用**；支持 `sparse_mode`、`actual_seq_lengths_kv` |
| `torch_npu.npu_incre_flash_attention` | torch_npu 算子 | Decode 专用（incre = incremental）；支持 paged KV (`block_table`, `block_size`) |
| `npu_fused_infer_attention_score` (V3) | aclnn 算子 | vLLM-Ascend 主用；统一 prefill+decode、支持 paged + RoPE 融合 |
| `kernels-ext-npu/flash-attn2` (HuggingFace) | Python wrapper | 把 transformers 里 `flash_attn` 的调用透明转发到 `npu_fusion_attention` |
| **FastAttention** (arXiv:2410.16663) | 学术开源（代码"will be available soon"） | FA2 → Ascend 的完整迁移论文，**two-level tiling / tiling-mask / tiling-AllReduce** 三大技术点公开了细节 |
| **AMLA** (arXiv:2509.25224) | 已并入 CANN | Ascend 910 上 614 TFLOPS，达到 86.8% 峰值，是 FlashMLA 的强力对手 |

**对 MInference 1.0 的意义**：
- `q_len == 1` 的 decoding 短路 → 直接换成 `npu_incre_flash_attention` 或 `npu_fused_infer_attention_score`。
- `starting_layer` 之前的若干 dense 层 → `npu_prompt_flash_attention` 或 `npu_fusion_attention`。
- `flash_attn_func`、`flash_attn_varlen_func` → `npu_fusion_attention`（input_layout="TND" + `actual_seq_qlen/kvlen`）。
- **零开发工作量**，只需要在 `minference_forward.py` 里加分支。

### 3.2 Block-Sparse FlashAttention

**现状：有两条可走的路径。**

| 路径 | 状态 | 评估 |
|---|---|---|
| `npu_fusion_attention` + `atten_mask`（mode 0 自定义 band，或 mode 6 prefix 压缩） | 官方 API 已生产可用 | 适合**静态稀疏 mask**；MInference 的 block-sparse 是 query-block 维度上的 top-k K-block，mask 是稀疏 0/1 矩阵——理论可用 mode 0 / 6 / 7 表示，但每个 query block 选不同 K block 集合的"行 ragged"模式可能要做 mask reshape |
| **TileLang `examples/blocksparse_attention/block_sparse_attn_tilelang.py`** | 已开源 | TileLang 作者明确说"相比 Dense FlashAttention 只需要多一个 `block_mask` 声明和一个 `if` 语句"。**几乎可以直接当 MInference block-sparse 的目标实现** |
| TileLang-Ascend (NPU 后端) | 已支持 attention 路径 | block-sparse 可以在 Ascend 后端跑，**这是最接近 MInference block-sparse 原始 kernel 的 NPU 替代方案** |

**对 MInference 1.0 的意义**：MInference 的 `_build_block_index`（Q/K 均池化 → topk K-block）是纯 PyTorch 代码，**与硬件无关**；只需要把 `_triton_block_sparse_attn_fwd_kernel` 换成 TileLang-Ascend 的 block-sparse 或 `npu_fusion_attention` + 动态构造的 `atten_mask`。

### 3.3 Vertical-Slash FlashAttention（主路径，最难也最有现成参考）

**现状：没有 1-to-1 的对应实现，但有非常接近的"祖先"。**

#### 关键发现：**TileLang-Ascend `examples/sparse_flash_attention/`**

这是本次调研中最重要的发现：

- 仓库：`tile-ai/tilelang-ascend`（2025-09-29 开源，A2/A3 已验证）
- 分支 `ascendc_pto` 下有完整文档 `bench_sfa/sparse_mla_performance_optimization_zh.md`
- 提供了三个相关算子：**FlashAttention / LightningIndexer / SparseFlashAttention**
- AscendC 参考实现来源于华为官方仓库 **cann-recipes-infer** 的 `npu_sparse_flash_attention.py`
- 评测结果：TileLang 编译的算子可达 AscendC 手写参考实现约 **0.90×** 的性能

虽然 TileLang-Ascend / cann-recipes-infer 的 SparseFlashAttention 是为 **DeepSeek-V3.2 Sparse Attention (DSA)** 设计的（pattern 是 "Lightning Indexer 选 top-k key block + sparse FA"），但与 MInference 的 vertical-slash 在**数据流上高度同构**：

| MInference 1.0（CUDA + Triton） | DSA / cann-recipes-infer（AscendC） |
|---|---|
| 在线估计 vertical / slash 索引（用最后 64 q × 全 K） | **Lightning Indexer**：小 GEMM 估计 token 重要性 |
| `convert_vertical_slash_indexes`（CUDA 索引展开） | top-k indexer + 拼装 K-block list |
| `_triton_mixed_sparse_attn_fwd_kernel`（两段 online softmax） | **Sparse Flash Attention** kernel |
| Triton（Python，相对易移植） | AscendC（C++）+ TileLang Python 两版都有 |

**对 MInference 1.0 的意义**：vertical-slash 的核心改造可以**直接基于 SparseFlashAttention 的代码骨架**来做，把"DSA 的 Lightning Indexer 输出"替换成"MInference 的 vertical + slash 索引输出"。这是迁移工作量最大的一块、也是最有现成参考的一块。

#### 备选方案

- **Triton-Ascend**：直接把 `pit_sparse_flash_attention_v2.py:48`（`_triton_mixed_sparse_attn_fwd_kernel`）的 Triton 代码搬过去，毕昇编译器走 Triton IR → AscendNPU IR → 二进制。**理论上是改造成本最低的路径**，但需要先验证社区报告的"80%–90% AscendC 性能"在两段 online softmax 场景是否仍然成立。Triton-Ascend 要求 CANN ≥ 8.3.RC1 / torch-npu ≥ 2.6.0.RC1。
- **AscendC 手写**：上限最高、开发量最大；只在前两条路径性能不够时再做。可以拿 FastAttention 论文的 two-level tiling 和 cann-recipes-infer 的 SFA 当起点。

### 3.4 Streaming（A-shape，sink + sliding window）

**现状：可以用 `npu_fusion_attention` + 自构造 mask 实现，无需自写 kernel。**

- `npu_fusion_attention` 的 `sparse_mode=0`（band）或 `sparse_mode=4`（band causal）即对应 A-shape 中"sliding window + causal"那部分。
- 前若干 sink token 可以用 `pre_tockens` / `next_tockens` 参数配合 atten_mask 的左侧拼接表示。
- 也可以通过 `prefix` 参数（mode=6 prefix 压缩）表达"始终可见的前缀"。

**对 MInference 1.0 的意义**：A-shape kernel 完全不必移植，**把 `streaming_forward` 改成一行 `torch_npu.npu_fusion_attention(..., sparse_mode=4, ...)` 即可**。

### 3.5 `convert_vertical_slash_indexes` CUDA 索引展开

**现状：无直接对应物，必须自写。但有接近的范式可参考。**

- 该 kernel 的特点：**dynamic shape + 双指针扫 + 分支多**，对昇腾的静态 tiling 模型不友好。
- 现成的相似工作：
  - vLLM-Ascend RFC #5507 中的 `hamming_dist_top_k` —— 也是 "K维上算分数 + topk + 输出索引" 的范式。
  - cann-recipes-infer 的 **Lightning Indexer** —— 也是 "estimator + topk + 索引列表" 范式。
  - vLLM-Ascend RFC #807 把 Python 的 `advance_step` 改写成 AscendC kernel —— 提供了 "把控制流密集的 Python 索引计算搬到 NPU" 的工程模板。
- 三个移植选项：
  1. **保留在 Host（CPU）端用 PyTorch / numpy 实现**（昇腾 Host CPU 多为 ARM Kunpeng，性能也不差）。开发量最小，但每步多一次 H2D 传输。
  2. **用 Triton-Ascend 重写**（Python 形式），让毕昇编译器自动处理 NPU 侧 tiling。中等开发量。
  3. **用 AscendC 重写**（参考 Lightning Indexer 的代码风格 + RFC #807 的工程模板）。开发量最大、性能最优。
- **建议**：第一版迁移走方案 1，跑通整套链路；性能瓶颈分析显示这一步占比高时再切到方案 2 / 3。

### 3.6 Per-head 调度循环（Python）

- 纯 Python，不涉及硬件，**直接复用** `minference_forward.py` 的循环。
- `best_pattern` 的 JSON（`configs/*.json`）也**直接复用**，文件格式无需变动。

---

## 4. 推理框架集成层的现状

无论选哪条 kernel 路径，最终都要接到一个推理框架。下面是几个候选宿主的现状：

### 4.1 vLLM-Ascend（最热门，社区最活跃）

- 上游仓库：`vllm-project/vllm-ascend`
- 最新动态：
  - **PR #7293**（v0.17.0）：FIA prefill 路径精简，直接构造 `out_list/lse_list` 喂给 `npu_attention_update`。
  - **RFC #5507 KVComp Sparse Attention**：明确把"长上下文 sparse attention"作为迁移目标，提出 hash 编码 + 自定义 `hamming_dist_top_k` + `reshape_and_cache_BNSD`。**MInference 的 vertical/slash 索引可以走类似插槽。**
  - **RFC #807** 把 `advance_step` 改写为 AscendC kernel；是"Python → AscendC custom op"的工程范本。
  - **RFC #2386** Automatic Kernel Fusion via `torch.fx.graph`（绕过 Inductor）。
  - DeepSeek-V3.2 Day 0 sparse attention 已在 vllm-ascend 实验性支持（参考 Red Hat Developer 文章）。
- **结论**：MInference 1.0 在 vLLM-Ascend 上的迁移大概率会以"新增一个 attention backend"或"扩展 KVComp RFC"的形式融入，**不要另起炉灶**。

### 4.2 MindIE（华为官方 LLM 推理 SDK）

- 闭源 SDK；优先在自家模型库（mindformers / MindSpeed-LLM）上发力。
- 长上下文 prefill 路径主要靠 context parallel（CP）+ FIA 算子。
- 公开材料中**未提到 MInference 风格的稀疏方案**；不建议作为第一宿主。

### 4.3 MindSpeed-LLM / AscendSpeed

- 训练框架（Megatron-LM 的 Ascend fork），主战场是训练加速。
- 长上下文训练侧已经有 sparse attention 相关研究在用（如 arXiv:2506.11498 的 Lag-Relative Sparse Attention 就基于 MindSpeed-LLM 实现）。
- 对纯推理迁移（MInference 仅加速 prefill）相关度低。

### 4.4 HF transformers + torch_npu

- 最轻量。只要 monkey-patch `LlamaAttention.forward`，把里面的 `flash_attn_func` 换成 `npu_fusion_attention`，把稀疏 kernel 换成上面 3.2/3.3 的实现即可。
- **建议**：第一版迁移就走这条路径，与 MInference 在 NVIDIA 上的 `patch_hf` 入口完全对称，把硬件差异收敛在 4 个 op 上。

---

## 5. 直接可用的开源仓库速查表

| 仓库 / 项目 | URL | 与 MInference 迁移的关系 |
|---|---|---|
| `Ascend/pytorch` (torch_npu) | https://github.com/Ascend/pytorch | PyTorch 适配层，所有 npu_* API 来源 |
| `kernels-ext-npu/flash-attn2` | https://huggingface.co/kernels-ext-npu/flash-attn2 | transformers 层 flash_attn 透明转发到 NPU |
| `Ascend/samples` | https://github.com/Ascend/samples | Ascend C 官方样例（含 Add / MatMul / Softmax 模板） |
| `tile-ai/tilelang-ascend` | https://github.com/tile-ai/tilelang-ascend | **FlashAttention / SparseFlashAttention / LightningIndexer 实现**；MInference vertical-slash 的最近邻参考 |
| `tile-ai/tilelang` (主仓) | https://github.com/tile-ai/tilelang | `examples/blocksparse_attention/`、`examples/flash_attention/` |
| `Ascend/triton-ascend` | https://gitcode.com/Ascend/triton-ascend（文档 `ascend.github.io/triton-ascend/`） | OpenAI Triton 的 NPU 后端；MInference 现有 Triton kernel 的迁移目标 |
| `chenqi123/cann-recipes-infer` | DeepWiki：deepwiki.com/chenqi123/cann-recipes-infer | DeepSeek-V3.2 DSA 的 NPU 参考实现，含 `npu_sparse_flash_attention.py` |
| `vllm-project/vllm-ascend` | https://github.com/vllm-project/vllm-ascend | 长上下文推理框架；RFC #5507 是 sparse attention 接入点 |
| `Ascend/MindSpeed-LLM` | https://gitee.com/ascend/MindSpeed-LLM | LLM 训练框架；sparse attention 研究的常用底座 |
| `Ascend/MindSpeed-MM` | https://gitee.com/ascend/MindSpeed-MM | 多模态 LLM 训练框架；附带模型迁移指南 |
| `Ascend/op-plugin` | https://gitee.com/ascend/op-plugin | npu_fused_infer_attention_score 等算子的源码 / 演进 |
| `Ascend/msit` | https://gitee.com/ascend/msit | 精度比对、性能 profile 工具链 |
| `Ascend/agent-skills` | https://github.com/Ascend/agent-skills | ATB / NNAL 安装等运维 skill |
| `Ascend/AscendNPU-IR` | https://gitee.com/ascend/ascendnpu-ir | 毕昇编译器对接第三方框架的中间层 |

---

## 6. 学术资料速查

| 论文 | arXiv | 与本迁移相关性 |
|---|---|---|
| **FastAttention: Extend FlashAttention2 to NPUs and Low-resource GPUs** | 2410.16663 | **直接迁移 FA2 → Ascend；披露 two-level tiling / tiling-mask / tiling-AllReduce** |
| **AMLA: MUL by ADD in FlashAttention Rescaling** | 2509.25224 | 已并入 CANN；Ascend 910 上达到 614 TFLOPS / 86.8% 峰值，Preload Pipeline + 层次化 tiling |
| **MInference 1.0** | 2407.02490 | 源算法本身 |
| **MMInference** | 2504.16083 | MInference 在 VLM 上的扩展，sparse pattern 复用 |
| **Native Sparse Attention (NSA)** | 2502.11089 | 硬件对齐 sparse attention 设计思路 |
| **Block Sparse Flash Attention (BSFA)** | 2512.07011 | 与 MInference 的 block-sparse 思路接近；drop-in replacement |
| **SeerAttention** | 2410.13276 | Learnable gate 选 block，可学习到 A-shape / vertical-slash |
| **Parallel Scan on Ascend AI Accelerators** | 2505.15112 | 昇腾上写 scan 类算子的范本，有助理解 MTE / Cube/Vector 协同 |
| **Automatic Ascend NPU Kernel Generation via DSL-Guided** | 2601.22760 | 自动生成 NPU kernel 的研究，前沿但未必能用 |
| **VSPrefill: Vertical-Slash Sparse Attention with Lightweight Indexing** | 2603.04460 | **后续工作，仍以 vertical-slash 为骨架；可了解上游算法演进** |
| **Context-Driven Performance Modeling for Causal Inference Operators on NPUs** | 2509.25155 | NPU 上 attention 算子的性能建模方法 |

---

## 7. 调研结论与下一步建议（不展开成方案，仅一句话陈述）

1. **没有任何团队公开发布过完整的 MInference 1.0 昇腾移植**，这是一块没人占的地。
2. **构成 MInference 1.0 的 4 个核心组件，每一个都有可复用的 NPU 资源**——dense FA、block-sparse FA、A-shape 三块基本"调 API"就能搞定；vertical-slash 这一最难的块，有 TileLang-Ascend `sparse_flash_attention` 例子 + cann-recipes-infer 的 `npu_sparse_flash_attention.py` 作为蓝本，再加 FastAttention 的论文级方法学指导。
3. **三条 NPU 算子开发路径**（Ascend C / Triton-Ascend / TileLang-Ascend）按"工程量 → 性能上限"排序，建议**先做 Triton-Ascend 全套 PoC**（与现有 Triton kernel 最对称），跑通后再决定哪几个算子值得用 AscendC / TileLang 改写。
4. **`convert_vertical_slash_indexes` 是唯一确认要自写的算子**；第一版可以保留在 CPU 端，待整体跑通后用 Triton-Ascend / AscendC 重写。
5. **框架集成首选 HF transformers + torch_npu**（与 MInference 现有 `patch_hf` 入口对称、最轻量）；后续视生产需要再接入 vLLM-Ascend（与 RFC #5507 / DeepSeek-V3.2 sparse attention 并轨）。

---

## 8. 引用与来源

### 8.1 Ascend C 与昇腾平台

- [Ascend C 自定义算子开发与使用指南 (MindSpore 2.3.1)](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.1/operation/op_custom_ascendc.html)
- [Overview-Quick Start-CANN Commercial Edition 8.0.0](https://www.hiascend.com/document/detail/en/canncommercial/800/quickstart/index/index.html)
- [Kernel Launch — Ascend C Operator Development (CANN 8.0.0)](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0056.html)
- [纯干货！一文 get 昇腾 Ascend C 编程入门全部知识点 (知乎)](https://zhuanlan.zhihu.com/p/653737107)
- [Ascend C 保姆级教程：我的第一份 Ascend C 代码](https://www.cnblogs.com/huaweiyun/p/17669701.html)
- [3 天上手 Ascend C 编程丨通过 Ascend C 编程范式实现一个算子实例](https://www.cnblogs.com/huaweiyun/p/17692967.html)
- [昇腾 Ascend C 编程入门教程（纯干货）](https://www.hiascend.com/developer/techArticles/20230830-1)
- [华为昇腾算子开发初级学习全攻略 (CSDN)](https://ascendai.csdn.net/691ef7380e4c466a32e99611.html)
- [从入门到实践：华为昇腾 Ascend C 算子开发指南 (CSDN)](https://ascendai.csdn.net/695fb7ceea53844658f573be.html)
- [Ascend NPU 硬件架构入门 (知乎)](https://zhuanlan.zhihu.com/p/3357780804)
- [Ascend C 高级算子优化技术：内存调度、并行策略与计算融合深度解析](https://ascendai.csdn.net/69d4d2d572111d255bf7dd7c.html)
- [【CANN 全新升级】毕昇编译器全新升级，开放 AscendNPU IR](https://www.hiascend.com/developer/techArticles/20250529-1)

### 8.2 Triton-Ascend / TileLang-Ascend

- [Triton Ascend (Huawei) NPU backend support · Issue #7855 · triton-lang/triton](https://github.com/triton-lang/triton/issues/7855)
- [编程指南 — Triton Ascend documentation](https://ascend.github.io/triton-ascend/sources/programming-guide/introduction.html)
- [昇腾 Triton-Ascend 开源实战 (CSDN)](https://ascendai.csdn.net/69396bca2087ae0db7a0c3ea.html)
- [Triton-Ascend 算子开发经验谈 (CSDN 鲲鹏昇腾)](https://hwcomputing.csdn.net/694d4932bf6b0e4b285e58ba.html)
- [tile-ai/tilelang-ascend (GitHub)](https://github.com/tile-ai/tilelang-ascend/)
- [tilelang-ascend SparseMLA Performance Optimization (中文)](https://github.com/tile-ai/tilelang-ascend/blob/ascendc_pto/examples/sparse_flash_attention/bench_sfa/sparse_mla_performance_optimization_zh.md)
- [tile-ai/tilelang (主仓 GitHub)](https://github.com/tile-ai/tilelang)

### 8.3 FlashAttention / Sparse Attention on Ascend

- [FastAttention: Extend FlashAttention2 to NPUs and Low-resource GPUs (arXiv 2410.16663)](https://arxiv.org/abs/2410.16663)
- [AMLA: MUL by ADD in FlashAttention Rescaling (arXiv 2509.25224)](https://arxiv.org/pdf/2509.25224)
- [chenqi123/cann-recipes-infer (DeepWiki)](https://deepwiki.com/chenqi123/cann-recipes-infer/1-overview)
- [Support flash-attention2 for Ascend NPU (HuggingFace kernels-ext-npu/flash-attn2)](https://huggingface.co/kernels-ext-npu/flash-attn2/commit/79153b4c8a7a9dd16a980b2da73afe9b305440c0)
- [torch_npu.npu_prompt_flash_attention (官方文档)](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000453.html)
- [torch_npu.npu_fusion_attention (官方文档)](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000139.html)
- [FlashAttentionScore (Ascend Pytorch 训练迁移)](https://www.hiascend.com/doc_center/source/zh/Pytorch/60RC1/ptmoddevg/trainingmigrguide/performance_tuning_0027.html)
- [torch_npu/meta/meta_registrations.py (Ascend pytorch on Gitee)](https://gitee.com/ascend/pytorch/blob/4be1afc321e08eb5fe174cf2e7956107755e56e0/torch_npu/meta/meta_registrations.py)

### 8.4 vLLM-Ascend & 推理框架

- [vllm-ascend Release Notes](https://docs.vllm.ai/projects/ascend/en/main/user_guide/release_notes.html)
- [RFC: KVComp Sparse Attention for Long-Context Inference (#5507)](https://github.com/vllm-project/vllm-ascend/issues/5507)
- [RFC: Custom Ascendc Kernel Of 'Prepare Input' (#807)](https://github.com/vllm-project/vllm-ascend/issues/807)
- [RFC: Automatic Kernel Fusion via torch.fx.graph (#2386)](https://github.com/vllm-project/vllm-ascend/issues/2386)
- [PR #7293: Simplify FIA prefill context merge path](https://github.com/vllm-project/vllm-ascend/pull/7293)
- [Context Parallel Guide (vllm-ascend)](https://docs.vllm.ai/projects/ascend/zh-cn/latest/user_guide/feature_guide/context_parallel.html)
- [DeepSeek-V3.2-Exp on vLLM, Day 0 (Red Hat Developer)](https://developers.redhat.com/articles/2025/10/03/deepseek-v32-exp-vllm-day-0-sparse-attention-long-context-inference)
- [[Usage] how to find ascend ops api (vllm-ascend #962)](https://github.com/vllm-project/vllm-ascend/issues/962)
- [[Feature, Hardware] add support for Ascend NPU (sglang #3781)](https://github.com/sgl-project/sglang/issues/3781)

### 8.5 训练框架与生态

- [Ascend/MindSpeed-LLM (Gitee)](https://gitee.com/ascend/MindSpeed-LLM)
- [Ascend/MindSpeed-MM (Gitee)](https://gitee.com/ascend/MindSpeed-MM)
- [Ascend Organization (GitHub)](https://github.com/ascend)
- [Ascend organization (Gitee)](https://gitee.com/ascend)
- [Ascend/agent-skills NNAL/ATB installer](https://github.com/Ascend/agent-skills/blob/master/skills/ascend-transformer-boost/skills/atb-nnal-installer/SKILL.md)
- [Ascend/op-plugin pull #2352](https://gitee.com/ascend/op-plugin/pulls/2352.patch?skip_mobile=true)
- [Ascend/AscendNPU-IR (Gitee)](https://gitee.com/ascend/ascendnpu-ir)
- [Ascend/msit (Gitee)](https://gitee.com/ascend/msit)
- [基于 Pytorch+ 昇腾 NPU 开发大模型指导 (CSDN)](https://blog.csdn.net/santanan/article/details/132394437)
- [Ascend 训练软件栈了解 (CSDN)](https://blog.csdn.net/m0_61864577/article/details/139506926)
- [昇腾 NPU 的踩坑之路 (知乎)](https://zhuanlan.zhihu.com/p/25147199560)

### 8.6 学术参考

- [MInference 1.0 (arXiv 2407.02490)](https://arxiv.org/pdf/2407.02490)
- [MMInference (arXiv 2504.16083)](https://arxiv.org/pdf/2504.16083)
- [Native Sparse Attention (arXiv 2502.11089)](https://arxiv.org/pdf/2502.11089)
- [Block Sparse Flash Attention (arXiv 2512.07011)](https://arxiv.org/abs/2512.07011)
- [SeerAttention (arXiv 2410.13276)](https://arxiv.org/pdf/2410.13276)
- [Parallel Scan on Ascend AI Accelerators (arXiv 2505.15112)](https://arxiv.org/html/2505.15112v1)
- [VSPrefill (arXiv 2603.04460)](https://arxiv.org/pdf/2603.04460)
- [Lag-Relative Sparse Attention In Long Context Training (arXiv 2506.11498)](https://arxiv.org/html/2506.11498v1)
- [Context-Driven Performance Modeling for Causal Inference Operators on NPUs (arXiv 2509.25155)](https://arxiv.org/pdf/2509.25155)
- [Automatic Ascend NPU Kernel Generation via DSL-Guided (arXiv 2601.22760)](https://arxiv.org/pdf/2601.22760)
