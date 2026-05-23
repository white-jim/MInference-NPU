# M1 — 上层 Python 链路打通（Dense Fallback）

> 阶段：M1（上层 Python 链路 / 不动 NPU kernel）
> 关联方案：`../../docs/migration_plan_v1.md` §3 M1
> 前置：M0 已完成（`docs/SETUP.md` + `tests/test_env.py` PASS）

---

## 1. 这一阶段做了什么

把整个 MInference 的"非 kernel"部分搬到 NPU 上跑通。三种稀疏分支
（vertical_and_slash / block_sparse / stream_llm）**全部退化为
`backend_npu.dense_attention`**（实际调用 `torch_npu.npu_fusion_attention`，
`sparse_mode=2` causal）。

这样做的意义：把所有"非 kernel 问题"（device 不一致 / dtype / RoPE / KV cache /
import 路径 / accelerate 多卡）先消干净，让 M2-M4 三个 milestone 只需关注 kernel
本身，每次只替换一个 kernel 分支就能验证回归。

---

## 2. 代码结构

```
minference/
├── __init__.py              # 顶层导出 MInference / patch_hf / 三个 sparse op 门面
├── models_patch.py          # MInference 类入口（v1：仅支持 attn_type ∈ {minference, dense, hf}）
├── patch.py                 # monkey-patch LlamaAttention.forward
├── minference_configuration.py
├── backend_npu/
│   ├── attention.py         # dense_attention / prefill_dense / decode_dense
│   └── cuda_shim.py         # convert_vertical_slash_indexes 占位（M4-a 实现）
├── modules/
│   ├── minference_forward.py     # NPU 版 per-head 调度 + 顶层 forward
│   └── minference_forward_upstream.py   # 上游原版（参考用，不 import）
├── ops/                     # 上游 sparse kernel（M2-M4 时复用骨架）
└── configs/                 # best_pattern JSON（直接复用上游）
```

**关键替换点**：

| 上游调用 | NPU v1 改造后 |
|---|---|
| `flash_attn_func(q, k, v, ..., causal=True)` | `backend_npu.dense_attention(q, k, v, causal=True)` |
| `flash_attn_func(q, k, v, ..., causal=False)` (q_len=1 decode) | `backend_npu.decode_dense(q, k, v)` |
| `vertical_slash_sparse_attention(q, k, v, v_idx, s_idx)` | `dense_attention(q, k, v, causal=True)` |
| `block_sparse_attention(q, k, v, topk)` | `dense_attention(q, k, v, causal=True)` |
| `streaming_forward(q, k, v, n_init, n_local)` | `dense_attention(q, k, v, causal=True)` |
| `from ..cuda import convert_vertical_slash_indexes` | `from ..backend_npu.cuda_shim import ...`（M1 抛 NotImplementedError，不会被触达） |
| `device="cuda"` 硬编码 | 跟随输入 tensor 的 device（device-agnostic） |

---

## 3. 跑法

### 3.1 端到端正确性测试（必跑）

```bash
cd MInference-NPU
python tests/test_dense_forward.py             # 标准
python tests/test_dense_forward.py -v          # 详细
```

包含 3 个子测试：

1. **test_dense_fallback_matches_eager** — 构造 tiny Llama（4 层 / 4 head /
   hidden=256），patch 后输出与 HF 原生 eager attention 差异 < 1e-2。
2. **test_minference_perhead_dense_fallback** — 临时生成 best_pattern JSON 让
   `attn_type="minference"` 走 per-head 调度，确认 per-head 循环骨架可跑且输出无 NaN。
3. **test_accelerate_device_map_auto** — `accelerate` 把模型按 layer 切到 ≥ 2 张
   NPU 上跑一次 forward，验证 device-agnostic 约束没破。**单卡环境自动 SKIP**。

预期：全 PASS，退出码 0。

### 3.2 真模型 smoke（可选）

```bash
# 单卡，ctx_len=8k
python examples/run_hf_minimal.py --model Qwen/Qwen2.5-7B-Instruct --ctx-len 8192

# 多卡 + 32k
python examples/run_hf_minimal.py --model meta-llama/Llama-3.1-8B-Instruct \
    --ctx-len 32768 --device-map auto

# 想验证"裸 dense"和"per-head 调度"的输出是否一致：
python examples/run_hf_minimal.py --attn-type dense --ctx-len 8192
python examples/run_hf_minimal.py --attn-type minference --ctx-len 8192
# v1 阶段两个输出应当 token-by-token 完全一致（M2-M4 替换后才会有差异）
```

---

## 4. 验收标准（与方案 §3 一致）

| 项 | 标准 | 验证 |
|---|---|---|
| 端到端 dense 链路能跑通 | tiny Llama forward 不挂 / 不 NaN / 不 OOM | `test_dense_fallback_matches_eager` PASS |
| 与 HF 原生 eager attention 数值一致 | max_abs_diff < 1e-2（fp16 噪声范围） | 同上 |
| per-head 调度骨架可跑 | best_pattern JSON 能加载、循环不挂 | `test_minference_perhead_dense_fallback` PASS |
| accelerate 多卡 device-agnostic | 2 张 NPU 上跑一次 forward 不挂（≥ 2 卡时） | `test_accelerate_device_map_auto` PASS |
| 真模型 32k prefill 能跑出文本 | 输出无乱码 / 连贯 | `examples/run_hf_minimal.py` 人眼对照 |

---

## 5. 排查（常见 FAIL 模式）

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `import minference` 报 `ModuleNotFoundError: torch_npu` | 不在 NPU 机器上，或 CANN 没 source | 见 `docs/SETUP.md` §3.1 |
| `dense_attention` 返回 NaN | dtype = fp32（NPU `npu_fusion_attention` 在 fp32 上不稳） | 模型加载时显式 `torch_dtype=torch.float16` |
| max_abs_diff > 1e-2 | softmax 数值噪声超出预期，或 head_dim 不是 2 的幂 | head_dim 建议 64 / 128；用 bf16 复跑对比 |
| accelerate 切层后挂 device mismatch | 有遗漏的 hard-code device | grep 仓库里所有 `.cuda()` / `device='npu:0'` |
| per-head 循环慢 | v1 阶段 dense 每 head 一次 npu_fusion_attention，是预期 —— M2-M4 替换为真稀疏 kernel 后会快回来 | 不必处理 |
| `convert_vertical_slash_indexes` NotImplementedError | 三种稀疏 fallback 没生效 | 检查 patch.py 是否真的把 forward 替换成了 NPU 版（看打印 "Patched ... .forward"） |

---

## 6. 与上游的对应关系（便于将来 PR 上游）

| 上游文件 | 本仓库 v1 改动 |
|---|---|
| `minference/patch.py:1313` (`patch_hf`) | `minference/patch.py:88` (`patch_hf`)，去掉 vLLM / inf_llm / kvcompress 分支 |
| `minference/patch.py:895` (`minference_patch`) | `minference/patch.py:55` (`minference_patch`)，仅保留 LlamaAttention.forward 替换 |
| `minference/modules/minference_forward.py:296` (`gather_last_q_vertical_slash_topk_v4`) | `minference/modules/minference_forward.py:159`，三 pattern dense fallback |
| `minference/modules/minference_forward.py:497` (`minference_forward`) | `minference/modules/minference_forward.py:194`，去掉 flash_attn 调用，改 backend_npu |
| `minference/csrc/vertical_slash_index.cu` | `minference/backend_npu/cuda_shim.py`（M1 占位 / M4-a 实现） |

上游版本作为 `*_upstream.py` 保留在仓库里，**仅供 M2-M4 对照，不应被 import**。

---

## 7. 下一步 → M2

M2 计划：把 `streaming_forward`（A-shape）从 dense fallback 替换为
`npu_fusion_attention(sparse_mode=4)` + sink 拼接。一行替换，热身用。

修改路径：
- `minference/__init__.py` 里的 `streaming_forward` → 指向新写的
  `minference/ops/streaming_kernel_npu.py:streaming_forward`
- 测试：`tests/test_streaming_kernel.py`（M2 时新增）
