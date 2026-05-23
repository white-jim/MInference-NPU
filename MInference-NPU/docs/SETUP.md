# MInference-NPU 环境安装与 M0 烟测

> 适用阶段：M0（环境与依赖准备）
> 关联方案：`../../docs/migration_plan_v1.md` §2 M0、`../../docs/context_checkpoint.md` §7
>
> ⚠️ 本文档假定目标服务器是华为昇腾 **910B 系列最常见配置**（详见 §1）。
> 实机版本可能微调，等用户实际盘下来再回头修订本节。

---

## 1. 假定的目标服务器配置（华为 910B 最常见）

| 维度 | 假定值 | 备注 |
|---|---|---|
| 服务器型号 | **Atlas 800T A2 训练服务器**（或推理同等型号 Atlas 800I A2） | 整机 8 卡，最主流 |
| NPU 型号 | **8 × 昇腾 910B3**（A2 架构） | 910B3 是 910B 系列里出货量最大的一档 |
| 单卡显存 | **64 GB HBM** | A2 标配 |
| 卡间互联 | HCCS 全互联 | 8 卡组内 |
| CPU | 鲲鹏 920（aarch64）或 Intel 至强（x86_64） | aarch64 更常见 |
| 系统内存 | ≥ 1 TB | 长上下文场景建议这一档 |
| 操作系统 | **Ubuntu 22.04 LTS**（aarch64）或 openEuler 22.03 LTS | 二选一，本计划默认 Ubuntu |

> 若实机不是 8 卡而是 4 卡 / 2 卡，本计划逻辑不受影响（v1 多卡支持靠 accelerate
> `device_map="auto"` 按 layer 切，对总卡数透明，参见 plan §0 / §7）。

---

## 2. 软件栈版本矩阵（v1 目标）

| 层 | 组件 | 选定版本 | 来源 | 约束理由 |
|---|---|---|---|---|
| 驱动 / 固件 | Ascend HDK | **24.1.RC3 及以上**（驱动 24.1.x） | 华为支持网站（昇腾社区） | 与 CANN 8.3.RC1 兼容 |
| 算子库 | **CANN Toolkit + Kernels** | **8.3.RC1** | 昇腾社区 | 方案约束 `≥ 8.3.RC1`（plan §2） |
| Python | CPython | **3.10.x** | conda / pyenv | torch 2.6 官方轮子覆盖；3.9 / 3.11 备选 |
| 深度学习框架 | PyTorch | **2.6.0** | PyPI | 与 torch_npu 2.6.0.RC1 锁版本 |
| NPU 适配 | **torch_npu** | **2.6.0.RC1** | 昇腾社区 wheel | 方案约束 `≥ 2.6.0.RC1` |
| 算子开发 | **triton-ascend** | 与 torch_npu 2.6 对齐的版本（pip 包名 `triton-ascend`） | gitee.com/ascend/triton-ascend 或华为昇腾社区 | v1 算子主路径 |
| 框架宿主 | transformers | **4.46.x**（≥ 4.45 即可） | PyPI | 与 patch_hf 改造目标对称（M1） |
| 多卡部署 | accelerate | **≥ 0.28**（取 0.34 稳定档） | PyPI | `device_map="auto"` 识别 npu:N |
| 辅助 | numpy / packaging / tqdm | 跟随上游 transformers 推荐 | PyPI | — |

**显式不安装 / 禁止安装**：
- `flash-attn` / `sgl_kernel` / `vllm_flash_attn` — NPU 上不可用，原代码所有相关 import 必须 `try-except` 守护并 fallback 到 `npu_fusion_attention`。
- `vllm` / `vllm-ascend` — v1 不集成。

---

## 3. 安装步骤（在目标 NPU 机器上执行）

> Windows 工作机仅用于编辑代码 / 写文档，**所有 NPU 测试必须在 Linux + Ascend 驱动的目标机上跑**。

### 3.1 系统级（root 一次性）

```bash
# 1. 装驱动 / 固件（Ascend HDK），按机型从昇腾社区下载对应 .run 包，安装后重启
# 2. 装 CANN 8.3.RC1（toolkit + kernels），按昇腾官方步骤一路 ./install.sh

# 3. 校验驱动 / 固件
npu-smi info
# 应看到 8 张 910B3，HBM 64GB

# 4. source CANN 环境（建议写进 ~/.bashrc）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 3.2 Python 虚拟环境

```bash
conda create -n minference-npu python=3.10 -y
conda activate minference-npu

# torch 2.6.0 + torch_npu 2.6.0.RC1（torch_npu 走昇腾社区 wheel，pip 默认源没有）
pip install torch==2.6.0
pip install torch_npu==2.6.0.RC1 \
    -i https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/pytorch/v2.6.0/   # 实际镜像以官方发布为准

# triton-ascend（与 torch_npu 2.6 配套版本）
pip install triton-ascend   # 若 pip 源没有，按 gitee.com/ascend/triton-ascend 编译指引装

# 其余依赖
pip install transformers>=4.45 accelerate>=0.28 numpy

# 装 minference-npu（本仓库）
cd D:/works/算法迁移/MInference-NPU   # 实际机器上路径会不同
pip install -e .
```

### 3.3 校验

```bash
# (a) torch_npu 可见 NPU
python -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
# 期望输出：True 8

# (b) triton-ascend 能 import
python -c "import triton; print(triton.__version__)"

# (c) MInference-NPU 包可导
python -c "import minference; print(minference.__version__)"
# 期望输出版本号（无异常）

# (d) 跑 M0 烟测
python tests/test_env.py
# 期望两个子测试都 PASS
```

---

## 4. test_env.py 用途

`tests/test_env.py` 是 M0 唯一的硬性验收脚本，包含两个独立子测试：

1. **`test_triton_ascend_vector_add`** — 跑一个最小 Triton-Ascend kernel（vector add），与 PyTorch CPU 参考结果做 `allclose`。验证 triton-ascend 工具链 / JIT 编译 / kernel launch 链路通。
2. **`test_npu_fusion_attention_smoke`** — 调 `torch_npu.npu_fusion_attention` 跑一个小尺寸 dense causal attention，与 PyTorch eager 实现（手写 softmax + mask）对比，差异容忍 `< 1e-2`（fp16 数值噪声）。验证 dense FA API 可用，为 M1 全 dense 链路铺路。

跑法：
```bash
python tests/test_env.py            # 直接跑，print PASS/FAIL
python tests/test_env.py -v         # 详细输出（差异、shape、device）
```

退出码：两子测试全 PASS 时退出码 0，任一 FAIL 退出码 1（便于 CI 接管）。

---

## 5. 常见踩坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `import torch_npu` 报 `libascendcl.so not found` | 没 `source set_env.sh` | 见 §3.1 step 4，写进 `~/.bashrc` |
| `npu-smi info` 看不到卡 | 驱动未装 / 固件不匹配 | 重装 Ascend HDK |
| triton kernel 编译报 `unsupported builtin` | triton-ascend 与 torch_npu 版本不匹配 | 查 gitee 仓库 README 的版本对应表 |
| `npu_fusion_attention` 返回 NaN | dtype 是 fp32 / 非 2 的幂 head_dim / mask shape 不对 | 走 bf16；head_dim 取 64/128；mask 用 `sparse_mode` 而非显式 mask |
| accelerate 切层后挂 device mismatch | 代码里有 hard-code `npu:0` / `tensor.cuda()` | 全部改成 device-agnostic（plan §0 约束 2） |

---

## 6. 等实机到位后要回填的项

`< TODO 替换 >` 标记的位置在实际部署后需要确认并改回具体值：

- §1 服务器型号 / NPU 型号（实机若不是 910B3 / 800T A2，照实改）
- §2 各组件具体小版本号（torch_npu / triton-ascend 拿到 wheel 之后回填精确版本）
- §3.2 torch_npu wheel 的实际下载 URL（昇腾社区当时的发布路径）
- `../../docs/context_checkpoint.md` §7「环境信息」段落同步更新

实机版本回填完成后，删除本文件顶部"假定"那行 ⚠️ 警告。
