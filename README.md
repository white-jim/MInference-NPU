# MInference-NPU

将微软 [MInference 1.0](https://github.com/microsoft/MInference) 的长上下文稀疏注意力推理加速方案，迁移到华为昇腾 NPU（Ascend 910B 系列）。

> **状态**：v1 早期阶段（M0/M1 链路打通中），尚未达到稀疏加速可用。详见 [`docs/migration_plan_v1.md`](docs/migration_plan_v1.md)。

---

## 仓库结构

```
.
├── MInference/          # 上游 microsoft/MInference 源码（vendored 副本，MIT）
├── MInference-NPU/      # 本项目：MInference 的 NPU 适配实现
│   ├── minference/      # NPU 版包代码
│   │   ├── backend_npu/ # torch_npu / triton-ascend 后端
│   │   ├── modules/     # patch 后的 attention forward
│   │   ├── ops/         # 稀疏算子（M2-M4 逐步替换）
│   │   └── configs/     # best_pattern JSON（沿用上游）
│   ├── tests/           # 烟测与回归测试
│   ├── examples/        # 最小可运行 HF demo
│   └── docs/            # 阶段性实现说明（M0、M1、…）
└── docs/                # 迁移方案与调研
    ├── ascend_migration_survey.md   # 昇腾生态调研
    ├── migration_plan_v1.md         # v1 迁移方案（拍板版）
    ├── MInference_1.0_implementation.md  # 上游实现拆解
    └── context_checkpoint.md        # 工作状态检查点
```

---

## v1 范围与约束

参见 [`docs/migration_plan_v1.md`](docs/migration_plan_v1.md)。要点：

- **算子主路径**：`triton-ascend`（与 `torch_npu` 2.6 配套），dense fallback 走 `torch_npu.npu_fusion_attention`
- **框架宿主**：HuggingFace transformers ≥ 4.45（暂不集成 vLLM）
- **稀疏分支**：v1 仅迁 `vertical_and_slash` / `block_sparse` / `stream_llm` 三种，其余（dilated / static / tri_shape / inf_llm / flexprefill / xattention）列入 v2
- **不依赖**：`flash-attn` / `sgl_kernel` / `vllm_flash_attn`（NPU 上不可用，import 处均加 try-except 守护）

里程碑：

| 阶段 | 目标 | 状态 |
|---|---|---|
| M0 | 环境与依赖（CANN 8.3.RC1 + torch_npu 2.6.0.RC1 + triton-ascend），`tests/test_env.py` PASS | 文档就绪，待实机验证 |
| M1 | 上层 Python 链路打通，三种稀疏分支全部退化为 dense fallback 跑通 HF 推理 | 代码框架就绪 |
| M2 | `stream_llm` 真稀疏 kernel（triton-ascend） | 未开始 |
| M3 | `block_sparse` 真稀疏 kernel | 未开始 |
| M4 | `vertical_and_slash` 真稀疏 kernel + 索引展开 | 未开始 |

---

## 快速开始

完整步骤见 [`MInference-NPU/docs/SETUP.md`](MInference-NPU/docs/SETUP.md)。简要：

```bash
# 在已装好 Ascend HDK + CANN 8.3.RC1 的目标机上
conda create -n minference-npu python=3.10 -y
conda activate minference-npu

pip install torch==2.6.0
pip install torch_npu==2.6.0.RC1 -i <昇腾社区 wheel 镜像>
pip install triton-ascend
pip install transformers>=4.45 accelerate>=0.28

cd MInference-NPU
pip install -e .

# 烟测
python tests/test_env.py
```

---

## 与上游 MInference 的关系

`MInference/` 目录是 microsoft/MInference 仓库的**完整源码副本**（vendored），便于和适配实现 diff 对照、且无需依赖网络拉取。本项目对上游代码**只读不改**；所有 NPU 适配新增/替换都集中在 `MInference-NPU/` 目录下。

- 上游版本快照：MInference 1.0（vendored at 2026-05-23）
- 上游协议：MIT（见 [`MInference/LICENSE`](MInference/LICENSE)）
- 上游论文 / 项目：<https://github.com/microsoft/MInference>

---

## 协议

- `MInference/` 子目录沿用上游 MIT 协议（版权归 Microsoft Corporation，见该目录下 `LICENSE`）
- 本仓库其余部分（`MInference-NPU/`、`docs/`、根目录文件）若无另行声明，按 MIT 发布
