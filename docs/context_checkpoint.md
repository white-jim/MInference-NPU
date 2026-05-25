# 上下文检查点 — MInference 1.0 → 昇腾 NPU 算法迁移

> 用途：在新会话中快速恢复工作上下文。详细信息在各专题文档里，本文件只列必读关键点。
> 最近更新：2026-05-25 — M0–M4 全部完成，两轮 bug 修复全部清完，下一步进 M5 实机联调。
> 工作目录：`D:\works\算法迁移\`（不是 git repo；子目录 `MInference-NPU/` 是 git repo）

---

## 1. 任务一句话

把 Microsoft [MInference 1.0](https://github.com/microsoft/MInference) 长上下文 LLM 推理加速算法迁移到 **华为昇腾 NPU**，v1 走 **Triton-Ascend + npu_fusion_attention** 路径，宿主 **HF transformers + torch_npu**。

- 上游源码（参考蓝本，**不动**）：`MInference/`
- v1 产出主线：`MInference-NPU/`
- 全部专题文档：`docs/`

---

## 2. v1 决策（已拍板，写在 `docs/migration_plan_v1.md`）

| 维度 | 选择 |
|---|---|
| 迁移范围 | vertical_and_slash + block_sparse + stream_llm + dense fallback |
| 排除项 | dilated / static / tri_shape / kvcompress / dist_ops / vLLM 集成 / FA3 / KV-CPU offload |
| 算子主路径 | Triton-Ascend + `npu_fusion_attention`（路径 A：bool mask） |
| `convert_vertical_slash_indexes` | CPU Python 双指针（M4-a）；NPU kernel 留 v2 |
| 框架宿主 | HF transformers + torch_npu；accelerate `device_map="auto"` 多卡 OK |
| 产出形态 | 独立 `MInference-NPU/` 完整 py 源 + tests + examples + docs + 独立 setup.py |

---

## 3. 进度

```
[x] 调研 + 方案 + 4 项决策拍板
[x] M0 环境与骨架                    详见 docs/SETUP.md
[x] M1 上层 Python 链路（dense fallback） 详见 docs/M1_dense_pipeline.md
[x] M2 Streaming kernel              详见 docs/M2_streaming.md
[x] M3 Block-Sparse kernel           详见 docs/M3_block_sparse.md
[x] M4 Vertical-Slash kernel（M4-a CPU 双指针 + M4-b NPU mask） 详见 docs/M4_vertical_slash.md
[x] Bug 修复 首轮（B1-B5/P3/P4，2026-05-24）
[x] Bug 修复 第二轮（B6-B9，2026-05-25）
[ ] M5 端到端联调 + 精度/性能/效果报告
[ ] docs/migration_v1_notes.md + docs/migration_v1_report.md
```

> 代码改造与实机校验解耦：Windows 工作机上写代码、syntax 检查通过即可推进；NPU 实机跑 `tests/test_*.py` 是用户拿到机器后并行做。

---

## 4. M5 入手要点（下一步直接接的事）

M5 不引入新代码，只做实机验证：

| 验证项 | 命令 |
|---|---|
| 环境烟测 | `python tests/test_env.py` |
| Dense pipeline | `python tests/test_dense_forward.py` |
| Streaming | `python tests/test_streaming_kernel.py` |
| Block-sparse | `python tests/test_block_sparse_kernel.py` |
| Vertical-slash | `python tests/test_vertical_slash_kernel.py` |
| HF 端到端 | `python examples/run_hf_minimal.py --attn_type minference ...` |

产出 `docs/migration_v1_notes.md`（实测踩坑）和 `docs/migration_v1_report.md`（精度/性能/效果三维对比）。

---

## 5. NPU v1 关键文件地图

| 角色 | 路径 |
|---|---|
| 顶层入口 | `MInference-NPU/minference/models_patch.py`（`MInference` 类） |
| HF monkey-patch | `MInference-NPU/minference/patch.py`（`minference_patch` / `patch_hf`） |
| Per-head 调度 + 顶层 forward | `MInference-NPU/minference/modules/minference_forward.py` |
| NPU dense 薄封装 | `MInference-NPU/minference/backend_npu/attention.py` |
| `convert_vertical_slash_indexes`（CPU Python） | `MInference-NPU/minference/backend_npu/cuda_shim.py` |
| Streaming kernel | `MInference-NPU/minference/ops/streaming_kernel_npu.py` |
| Block-Sparse kernel | `MInference-NPU/minference/ops/block_sparse_kernel_npu.py` |
| Vertical-Slash kernel | `MInference-NPU/minference/ops/vertical_slash_kernel_npu.py` |
| 上游参考（不 import，仅对照） | `MInference-NPU/minference/{patch,models_patch}_upstream.py` + `modules/minference_forward_upstream.py` |
| Best-pattern JSON | `MInference-NPU/minference/configs/*.json`（直接复用上游） |

上游对照路径（grep `MInference/` 子树即可，不在此处重复）。

---

## 6. MInference 1.0 算法关键点（迁移时反复用到）

详见 `docs/MInference_1.0_implementation.md`。**只摘必须记住的事实**：

- **加速对象**：仅 prefill；`q_len == 1` decode 短路成 dense flash_attn。
- **三种稀疏**：`vertical_and_slash`（主力，~95% head）、`block_sparse`、`stream_llm`（A-shape）。
- **best_pattern JSON**：每 (layer, head) 一组 `(pattern_type, v_size, s_size, score)`；常见默认 `vertical_and_slash, v=1000, s=6096`。
- **在线估计**（vertical_and_slash）：用最后 64 个 query 与全 K 做 QK，列和→topk→`v_idx`（升序）；反对角线和→topk→`s_idx`（降序）。强制保留前 30 列 sink + 最近 100 条 slash。
- **per-head 循环**：不同 head 选不同 pattern，无法 batch；本身开销远小于稀疏 kernel 收益。
- **starting_layer**：前若干层走 dense（pattern 未收敛）。
- **v1 不构建** 上游 `minference.cuda` CUDAExtension（`csrc/{kernels.cpp, vertical_slash_index.cu}`），由 `backend_npu/cuda_shim.py` 顶替。

---

## 7. 环境与守则

- **平台**：Windows 10 Pro / bash / Unix 风格路径；目录含中文「算法迁移」，shell 操作需引号包裹。
- **目标 NPU 服务器**（实机）：aarch64 / openEuler 22.03 LTS-SP4 / 910B3 / 驱动 25.0.rc1.1 / CANN Toolkit 8.1.RC1（`ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1`）/ Python 3.10 / torch 2.5.1 + torch_npu 2.5.1 / transformers 4.57.3 / accelerate 0.34.2。Triton-Ascend 在 CANN 8.1.RC1 下不作为默认依赖。完整矩阵见 `MInference-NPU/docs/SETUP.md`。
- **device-agnostic 守则**（M1 落地、M2-M4 守住）：
  1. 禁止写死 `npu:0` 或 `device="cuda"`，所有 tensor 跟随输入 device。
  2. Triton-Ascend kernel launch 也跟随输入 device。
  3. accelerate `device_map="auto"` 多卡靠这条覆盖；vLLM TP / 序列并行 / EP 留 v2。
- **长上下文多卡**：手动给 `max_memory` 留 KV 余量（配置层面，不改代码）。

---

## 8. 仍待处理（性能优化，留 M5 实测后做）

以下问题影响性能而非正确性，重构成本大且容易引入新错，**先看 M5 真实 profile 数据再判断要不要动**：

| 编号 | 文件 | 一句话 |
|---|---|---|
| P5 | `ops/vertical_slash_kernel_npu.py` + `backend_npu/cuda_shim.py` | vertical/slash 的 CPU Python 双指针 index convert 在 128k 上可能秒级；待向量化或 NPU kernel |
| P6 | M3/M4 kernel | token 级 `[B,H,S_q,S_k]` bool mask 16k 就重，32k+ 不实用；需真正的 block/index 稀疏 kernel |
| P7 | M3/M4 kernel | `S > 16384` silent dense fallback；M5 性能测试时若覆盖长上下文需手动检查 warning，避免误判 |
| P8 | `modules/minference_forward.py` | per-head 循环造成大量小 kernel launch；按 pattern 分组 batching 是优化空间 |
| P9 | `modules/minference_forward.py` | GQA/MQA 用 `repeat_kv` 物理复制，长上下文放大 KV 工作量 |

性能修复决策原则：M5 拿到真实瓶颈再动，不预先重构。
