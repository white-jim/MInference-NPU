# MInference-NPU 环境安装与 M0 烟测

> 适用阶段：M0（环境与依赖准备）
> 关联方案：`../../docs/migration_plan_v1.md` §2 M0、`../../docs/context_checkpoint.md` §7
>
> 本文档已按当前实机环境修订：openEuler 22.03 LTS-SP4 / 910B3 /
> 驱动 25.0.rc1.1 / CANN Toolkit 8.1.RC1。

---

## 1. 目标服务器配置

| 维度 | 实机值 | 备注 |
|---|---|---|
| NPU 型号 | **昇腾 910B3** | 以 `npu-smi info` 为准 |
| CPU 架构 | **aarch64** | 选择 aarch64 wheel |
| 操作系统 | **openEuler 22.03 LTS-SP4** | 实机环境 |
| 驱动 / npu-smi | **25.0.rc1.1** | 实机环境 |
| CANN Toolkit | **8.1.RC1** | `ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1` |

> 若实机不是 8 卡而是 4 卡 / 2 卡，本计划逻辑不受影响（v1 多卡支持靠 accelerate
> `device_map="auto"` 按 layer 切，对总卡数透明，参见 plan §0 / §7）。

---

## 2. 软件栈版本矩阵（v1 目标）

| 层 | 组件 | 选定版本 | 来源 | 约束理由 |
|---|---|---|---|---|
| 驱动 / 固件 | Ascend HDK | **25.0.rc1.1** | 实机已安装 | 与当前 CANN 8.1.RC1 环境配套 |
| 算子库 | **CANN Toolkit + Kernels** | **8.1.RC1** | 实机已安装 | 当前机器固定版本 |
| Python | CPython | **3.10.x** | conda / pyenv | torch_npu 2.5.1 提供 cp310 aarch64 wheel |
| 深度学习框架 | PyTorch | **2.5.1** | PyPI | CANN 8.1.RC1 官方匹配项中最接近原计划 |
| NPU 适配 | **torch_npu** | **2.5.1** | 昇腾社区 wheel | 与 PyTorch 2.5.1 / CANN 8.1.RC1 匹配 |
| 算子开发 | **triton-ascend** | 默认不安装 | - | 当前 Triton-Ascend 文档要求 CANN 8.2+/torch_npu 2.6 线，CANN 8.1 下不作为 v1 必需项 |
| 框架宿主 | transformers | **4.57.3** | PyPI | 对齐项目 requirements |
| 多卡部署 | accelerate | **0.34.2** | PyPI | 对齐项目 requirements |
| 辅助 | numpy / packaging / tqdm | 见 `requirements.txt` | PyPI | 对齐项目 requirements |

**显式不安装 / 禁止安装**：
- `flash-attn` / `sgl_kernel` / `vllm_flash_attn` — NPU 上不可用，原代码所有相关 import 必须 `try-except` 守护并 fallback 到 `npu_fusion_attention`。
- `vllm` / `vllm-ascend` — v1 不集成。

---

## 3. 安装步骤（在目标 NPU 机器上执行）

> Windows 工作机仅用于编辑代码 / 写文档，**所有 NPU 测试必须在 Linux + Ascend 驱动的目标机上跑**。

### 3.1 系统级（root 一次性）

```bash
# 1. 装驱动 / 固件（Ascend HDK），按机型从昇腾社区下载对应 .run 包，安装后重启
# 2. 确认 CANN 8.1.RC1（toolkit + kernels）已安装

# 3. 校验驱动 / 固件
npu-smi info
# 应看到 8 张 910B3，HBM 64GB

# 4. source CANN 环境（建议写进 ~/.bashrc）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source 后 ASCEND_HOME_PATH 会指向 /usr/local/Ascend/ascend-toolkit/8.1.RC1
```

### 3.2 Python 虚拟环境

```bash
conda create -n minference-npu python=3.10 -y
conda activate minference-npu

# Python 依赖。torch_npu 需要使用与 CANN 8.1.RC1 匹配的 aarch64 wheel。
pip install -r requirements.txt

# 装 minference-npu（本仓库）
pip install -e .
```

### 3.3 校验

```bash
# (a) torch_npu 可见 NPU
python -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
# 期望输出：True 8

# (b) MInference-NPU 包可导
python -c "import minference; print(minference.__version__)"
# 期望输出版本号（无异常）

# (c) 跑 M0 烟测
python tests/test_env.py
# 期望 npu_fusion_attention 子测试 PASS

# Triton-Ascend 不是 CANN 8.1.RC1 默认项。若后续升级到 CANN 8.2+/torch_npu 2.6 线，
# 可额外执行：
python tests/test_env.py --with-triton
```

---

## 4. test_env.py 用途

`tests/test_env.py` 是 M0 的硬性验收脚本，默认包含当前实机必需子测试：

1. **`test_npu_fusion_attention_smoke`** — 调 `torch_npu.npu_fusion_attention` 跑一个小尺寸 dense causal attention，与 PyTorch eager 实现（手写 softmax + mask）对比，差异容忍 `< 1e-2`（fp16 数值噪声）。验证 dense FA API 可用，为 M1 全 dense 链路铺路。

可选：
- **`test_triton_ascend_vector_add`**（`--with-triton`）— 跑一个最小 Triton-Ascend kernel（vector add）。CANN 8.1.RC1 默认不要求安装 Triton-Ascend。

跑法：
```bash
python tests/test_env.py            # 默认烟测，print PASS/FAIL
python tests/test_env.py -v         # 详细输出（差异、shape、device）
python tests/test_env.py --with-triton -v
```

退出码：已启用的测试全 PASS 时退出码 0，任一 FAIL 退出码 1（便于 CI 接管）。

---

## 5. 常见踩坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `import torch_npu` 报 `libascendcl.so not found` | 没 `source set_env.sh` | 见 §3.1 step 4，写进 `~/.bashrc` |
| `npu-smi info` 看不到卡 | 驱动未装 / 固件不匹配 | 重装 Ascend HDK |
| triton kernel 编译报 `unsupported builtin` | triton-ascend 与 torch_npu / CANN 版本不匹配 | CANN 8.1.RC1 下先不要安装/启用 Triton-Ascend；升级到匹配矩阵后再测 |
| `npu_fusion_attention` 返回 NaN | dtype 是 fp32 / 非 2 的幂 head_dim | 走 bf16/fp16；head_dim 取 64/128 |
| `test_npu_fusion_attention_smoke` 报 `max_abs_diff≈3.6`（≈full attention） | **CANN 8.1.RC1 + torch_npu 2.5.1 下 `sparse_mode=0/2` 不传 `atten_mask` 都退化为 full attention**，即使 `sparse_mode=2` 标称 causal | causal 路径统一改成 `sparse_mode=1 + 显式 bool atten_mask`（`True=masked`，NPU 惯例）。已在 `backend_npu/attention.py` 与 `tests/test_env.py` 落地，详见下方 §5.1。 |
| accelerate 切层后挂 device mismatch | 代码里有 hard-code `npu:0` / `tensor.cuda()` | 全部改成 device-agnostic（plan §0 约束 2） |

### 5.1 sparse_mode causal 语义在 CANN 8.1.RC1 的实测结论

实测组合（B=1, N=4, S=256, D=128, fp16）：

| 调用 | 与手写 PyTorch eager causal 参考的 `max_abs_diff` | 实际语义 |
|---|---|---|
| `sparse_mode=0`（无 mask） | ~3.6 | full attention |
| `sparse_mode=2`（无 mask） | ~3.6 | full attention（非 causal） |
| `sparse_mode=2 + pre_tockens/next_tockens` | ~3.6 | full attention |
| **`sparse_mode=1 + 显式 [S_q, S_k] bool atten_mask`** | **~0.00195（mean~2e-5）** | **正确 causal** |

结论：路径 A 的所有 dense / 稀疏 causal 调用都走 `sparse_mode=1 + 显式 atten_mask`。
- bool mask 约定：`True = masked out`（与 M3/M4 的 `_block_sparse_npu` / `_vertical_slash_npu` 保持一致）。
- 个别 torch_npu 小版本只接受 `uint8` mask，已经在调用点用 `try/except TypeError` 兜底转换。
- 该结论待 CANN 升级到 8.2+/torch_npu 2.6 线时复测；若届时 `sparse_mode=2` 行为修正，可回滚到无 mask 写法以省 `[S_q, S_k]` 显存。

---

## 6. 等实机到位后要回填的项

`< TODO 替换 >` 标记的位置在实际部署后需要确认并改回具体值：

- 若后续升级 CANN 到 8.2/8.3 线，再重新评估 Triton-Ascend 与 torch_npu 版本。
- `../../docs/context_checkpoint.md` §7「环境信息」段落需与本文件保持同步。
