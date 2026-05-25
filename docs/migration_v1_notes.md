# MInference 1.0 → 昇腾 NPU 迁移踩坑笔记 v1

> 用途：把 v1 实施过程中（含实机联调阶段）踩到的所有坑，按"现象 → 根因 → 解法 → 是否修复"的格式集中记录，供后续 v2 / 别的 NPU 适配项目复用。
> 创建日期：2026-05-25
> 关联文档：`docs/context_checkpoint.md`（实时进度）、`docs/migration_v1_report.md`（精度/性能/效果三方对比报告）、`MInference-NPU/docs/SETUP.md`（环境与依赖）

---

## 0. 阅读指南

- 每条坑都附**触发条件**、**根因**、**修复位置（文件:行）**、**是否已修**。
- "已修"=v1 内代码改动可绕开；"v1 设计代价"=不能在 v1 内绕开，需 v2 处理；"环境侧"=外部依赖触发，代码层只能配合。
- 按踩坑发生的时间顺序排列，便于结合 `context_checkpoint.md` §8/§9 时间线对照。

---

## 1. CANN 8.1.RC1 + torch_npu 2.5.1 与 triton-ascend 版本矩阵不匹配（环境侧）

**触发**：M0 实机检查发现服务器 CANN = 8.1.RC1、torch_npu = 2.5.1，而 `triton-ascend` 要求 CANN ≥ 8.2 + torch_npu ≥ 2.6（见 `MInference-NPU/docs/SETUP.md` §2 版本矩阵）。

**根因**：CANN 与 triton-ascend 是强耦合的版本三角，跨大版本不可向下兼容。

**影响**：原 v1 立项方案（`docs/migration_plan_v1.md` §0 决策 2）拍板 Triton-Ascend 作为主路径，需作废，改走 **`npu_fusion_attention` + 显式 `atten_mask`**（路径 A）。

**修复**：方案变更已回填到 `migration_plan_v1.md` 顶部"⚠️ 实施路径变更说明"块，并在 `context_checkpoint.md` §8（P5/P6）落档。**Triton-Ascend 降级为 v2 备选**（路径 B）。

**状态**：v1 设计已绕开；CANN 升级到 8.2+ 之后 v2 可重启路径 B。

---

## 2. `torch_npu.npu_fusion_attention` 的 `sparse_mode` 在 CANN 8.1.RC1 下不按文档语义生效（环境侧 + v1 设计代价）

**触发**：M5 实机启动时 `tests/test_env.py::test_npu_fusion_attention_smoke` FAIL（`max_abs_diff=3.633`），定位发现 `sparse_mode=2` 没做 causal mask；M5 streaming kernel 复测时 `sparse_mode=4` 同款失败（NPU 三 case `max_abs_diff` 0.13~0.40）。

**实测 sparse_mode 行为表**：

| 调用 | 文档语义 | 实测语义 |
|---|---|---|
| `sparse_mode=0`（无 mask） | full attention | ✅ full attention |
| `sparse_mode=2`（无 mask） | causal | ❌ **退化为 full attention** |
| `sparse_mode=2 + pre_tockens/next_tockens` | causal + window | ❌ full attention |
| `sparse_mode=4 + pre_tockens=n_local-1, next_tockens=0` | band / sliding-window | ❌ **退化为 full attention** |
| `sparse_mode=1 + 显式 [S_q,S_k] bool atten_mask`（True=masked） | 用户自定义 mask | ✅ 行为正确（vs eager ref `max_abs_diff ~0.00195`） |

**根因**：CANN 8.1.RC1 + torch_npu 2.5.1 组合下，`npu_fusion_attention` 的内置稀疏模式只有 mode 1 真的会读 `atten_mask`，mode 2/4 等都被静默退化。

**解法（已采纳）**：v1 的所有 causal / band 调用统一改成 **`sparse_mode=1 + 显式 bool atten_mask`**，True=masked，与 M3/M4 kernel 既有约定一致。

**修复位置**：
- `MInference-NPU/minference/backend_npu/attention.py::dense_attention` —— causal mask 显式构造，调用走 sparse_mode=1
- `MInference-NPU/tests/test_env.py::test_npu_fusion_attention_smoke` —— smoke 测试同步改造
- `MInference-NPU/minference/ops/streaming_kernel_npu.py::_streaming_npu` —— 段 1 由 `sparse_mode=4` 改 `sparse_mode=1 + 显式 sliding-window bool mask`
- 调用点统一用 `try/except TypeError` 兜底 `uint8` mask（部分 torch_npu 小版本只接受 uint8）

**实测结果**：
- `test_env.py::test_npu_fusion_attention_smoke` fp16 `max_abs_diff=1.953e-03` PASS
- `test_streaming_kernel.py` 32/32（含 3 NPU case）PASS

**状态**：v1 已修；CANN 升级到 8.2+ 时复测 sparse_mode=2/4，若行为修正可回滚省 `[S_q, S_k]` 显存。

**关联 memory**：`project_minference_npu_fa_sparse_mode_quirk`。

---

## 3. 显式 bool mask 的 `O(S²)` 显存代价（v1 设计代价）

**触发**：M5 末轮 128K 多卡 `device_map=auto` minference 跑 OOM：

```
backend_npu/attention.py:131
causal_mask = torch.ones(s_q, s_k, device=q.device, dtype=torch.bool).triu(...)
RuntimeError: NPU out of memory. Tried to allocate 16.00 GiB
```

`131072 × 131072` bool = **17 GiB 仅是 mask 本身**，加上 attention 中间激活就把 60 GiB 单卡吃满。

**根因**：直接由 §2 的解法链生出来的代价——既然走 `sparse_mode=1` + 显式 mask，那 mask 就必须是 `[1,1,S_q,S_k]` 物化张量，显存 O(S²)。

**估算上限**（910B3 60 GiB HBM）：

| ctx | mask 显存 | 是否可跑（单卡） |
|---|---|---|
| 8 K | 64 MiB | ✅ |
| 16 K | 256 MiB | ✅ |
| 32 K | 1 GiB | ✅ |
| 64 K | 4 GiB | ⚠️ 接近边界 |
| 128 K | 17 GiB | ❌ OOM |

**v1 内已采用的缓解**：vertical-slash 路径在 `S > 16384` 时由 P7 silent fallback 到 dense（见 `minference_forward.py:246` UserWarning）。但 dense 本身也是 §2 的 sparse_mode=1 + 显式 mask 路径，所以 fallback 救不了 128K。

**v2 出路**：
1. CANN 升级后 `sparse_mode=2` 修复，省掉 `[S, S]` mask（首选）；
2. 切换到 Triton-Ascend 真稀疏 kernel，attention 内部分块、不物化整张 mask；
3. dense fallback 自身分块（chunk-by-chunk attention）。

**状态**：v1 设计代价，已通过 `>16384 silent fallback` 缓解但未根治；写入 `migration_v1_report.md` 的"已知限制"。

---

## 4. transformers 4.57.3 LlamaAttention forward 签名变更（已修）

**触发**：M5 启动 HF 端到端时，`minference_forward` 的旧签名与 transformers 4.57.3 `LlamaDecoderLayer` 的 kwargs 调用不兼容，报 `TypeError: forward() got unexpected keyword argument 'position_embeddings'`。

**变更点（4.45 → 4.57）**：

| 维度 | 旧 | 新 |
|---|---|---|
| `past_key_value` 命名 | 单数 | 复数 `past_key_values` |
| `output_attentions` | 必传 | 已删除 |
| `position_embeddings`(cos,sin) | 由 attention 内部 `self.rotary_emb` 算 | 由 decoder layer 传入 |
| `num_heads` / `num_key_value_heads` | 挂在 attention module 上 | 仅在 `self.config` |
| 返回值 | `(attn_output, attn_weights, past_key_value)` | `(attn_output, attn_weights)` |

**解法**：`minference_forward.py::forward` 重写签名，全 kwargs 带默认值，新老路径都兼容。

**修复位置**：`MInference-NPU/minference/modules/minference_forward.py:284-450`（带详细 docstring 说明 4 处迁移点）。

**状态**：已修，`test_env.py` + `test_dense_forward.py` + HF 端到端三轮 minference 都 PASS。

---

## 5. streaming kernel 测试用例 `s_q > s_k` 引发 nan（用例 bug，非 kernel bug）

**触发**：`test_streaming_kernel.py::test_streaming_forward_vs_naive[short_circuit_klen_lt_nlocal-f32/f16]` `max_abs_diff=nan`。

**根因**：用例 `(n_local=512, s_q=256, s_k=128)` 中 `s_q > s_k`。在 causal masking 下 `abs_i = s_k - s_q + i`，前 `(s_q - s_k)` 个 query 一个 key 都看不到，softmax(-inf) = nan。属于**用例本身不合法**（prefill 场景应 `s_q == s_k`），非被测代码的 bug。

**解法**：用例改成 `(64, 512, 128, 128, "short_circuit_klen_lt_nlocal")` —— 仍满足 `s_k=128 < n_local=512` 的短路条件，但 `s_q == s_k` 合法。

**修复位置**：`MInference-NPU/tests/test_streaming_kernel.py` `_STREAMING_CASES`，并加注释解释为什么这条用例必须 `s_q == s_k`。

**状态**：已修，32/32 PASS。

---

## 6. kernel 测试文件无 `__main__` 入口（非 bug，约定即可）

**触发**：用户在服务器跑 `python tests/test_streaming_kernel.py`，输出只有 Ascend toolkit 的 owner 警告，没有任何测试结果。

**根因**：`test_streaming_kernel.py` / `test_block_sparse_kernel.py` / `test_vertical_slash_kernel.py` 使用 pytest 装饰器风格（fixture + parametrize），**不带 `if __name__ == "__main__": pytest.main([__file__])` 入口**。直接 `python file.py` 不会触发 pytest，只是把模块 import 进来什么也不做。

**对比**：`test_env.py` / `test_dense_forward.py` 是手写函数 + main 入口，可直接 `python file.py`。

**解法**：约定 kernel 类测试统一用 `python -m pytest tests/test_*.py -v` 调用，文档明确。

**状态**：约定即可，未改文件（用户确认不动）。

---

## 7. NPU `npu-smi info` AICore 0% 而显存非空 ≠ CPU 推理（认知校准）

**触发**：M5 第二轮 16K minference 跑期间，用户观察到 NPU 显存占用 24 GiB，但 `npu-smi info` AICore 持续显示 0%、偶尔窜到 1%，怀疑是不是退化到 CPU 推理。

**排除"CPU 推理"假设的证据**：
1. HBM 占用 24 GiB ≈ 8B Llama fp16 权重 16 GiB + KV cache + 激活，与 NPU 加载一致；CPU 推理时 HBM 应为 0；
2. dense 路径 16K 仅 4.57s，CPU 不可能；
3. dense 同环境下 AICore 实测 80%+（用户验证）。

**真实原因**：AICore 0% 反映 NPU **在等 host 端 Python**。`npu-smi info` 采样窗口很容易打在 host 端循环 / 索引构造 / mask 构建的间隙；如果 host overhead 远大于 device compute，AICore 会持续接近 0。

**根本根因**：见下文 §8，是 minference per-head Python 循环造成的 host-bound。

**解法**：这不是 bug，是 v1 架构代价的可观测信号。已在 `migration_v1_report.md` 性能章节解释。

**状态**：认知层校准，无需代码改动。

---

## 8. 【v1 最大设计代价】minference per-head Python 循环 host-bound

**触发**：M5 第二轮三组数据：

| ctx | attn | 时间 | 输出 | NPU AICore |
|---|---|---|---|---|
| 8K | minference | 308.53s | 正常 | ~0% |
| 16K | minference | 616.32s | 正常 | ~0% |
| 16K | dense | **4.57s** | **与 minference 一字不差** | ~80% |

**现象**：minference 路径比 dense 慢 **~135 倍**，输出 bit-identical，NPU 算力实测空转。

**根因（代码定位）**：`MInference-NPU/minference/modules/minference_forward.py:413-437`

```python
if q_len != 1:
    if self._minference_attn_type == "dense":
        output = dense_attention(query_states, key_states, value_states, causal=True)  # 一次大调用，全 head
    else:
        output = torch.empty_like(query_states)
        for head in range(query_states.size(1)):              # ← 32 个 head 的 Python 循环
            q = query_states[:, head, :, :].unsqueeze(1)
            k = key_states[:, head, :, :].unsqueeze(1)
            v = value_states[:, head, :, :].unsqueeze(1)
            ...
            attn_output = gather_last_q_vertical_slash_topk_v4(self, q, k, v, head)  # 内部 ≥7 个 NPU 算子调用
            output[:, head : head + 1] = attn_output
```

每次 prefill 的算子调用规模：

| 路径 | 大算子调用数 | 单调用规模 | 备注 |
|---|---|---|---|
| **dense** | 32 次 `npu_fusion_attention` | B=1, H=32, S=16384 | 单次大算子，NPU 算到饱和 |
| **minference** | 32 层 × 32 head × （QK 估计 + softmax + topk×2 + sort×2 + bool mask 构造 + `npu_fusion_attention`）≈ **7000+ 次小算子** | B=1, **H=1**, S=16384 | NPU 几十微秒算完一个就等 Python 派下一个 |

**为什么 GPU 上没这个问题**：
1. CUDA stream + 异步 launch，host launch latency 在 microseconds 量级；NPU 经验值比这大一个量级以上，1024 次循环就堆成秒级。
2. 上游 `vertical_slash_sparse_attention_triton` 把 per-head 调度 + topk-index 全融进单个 Triton kernel，host 只调 1 次。NPU 上没有等价物（`npu_fusion_attention` 是粗粒度黑盒），per-head 调度只能上 Python 循环。

**是否 bug**：**不是**，是 v1 立项时即拍板的设计代价（`migration_plan_v1.md` §0 决策：v1 只追"算法跑通 + 精度对齐"，不追性能）。这次实测把代价**量化**了。

**v2 必做的第一项**：消除 per-head Python 循环，三种可选路径：
1. **batched 调度**（首选）：32 个 head 的 v_idx / s_idx 一次性并行构造，单次 batched mask + 单次全 head `npu_fusion_attention`；
2. Triton-Ascend kernel（等环境）：把 per-head 调度融进单个 kernel；
3. NPU 图模式（`torch_npu.npu.graph_mode`）capture 静态图，消除 launch overhead。

**状态**：v1 设计代价、未在 v1 内修复；v2 第一优先级。

---

## 9. vertical-slash 在 `S > 16384` silent fallback 到 dense（v1 设计代价）

**触发**：M5 32K minference run，stdout 含

```
minference_forward.py:246 UserWarning:
vertical_slash_sparse_attention: 序列长度 32768 > 16384。
bool mask 构建需 O(S²) 内存，退化为 causal dense。
```

**根因**：vertical-slash 真稀疏路径要构 `[1,1,S,S]` bool mask 表达 vertical 列 + slash 对角线；`S > 16384` 时这张 mask 单张 ≥ 1 GiB，叠加多 head 显存爆。v1 选择 silent fallback 到 dense（P7 兜底）。

**影响**：v1 的 minference 真稀疏路径有效区间 = `S ≤ 16384`；`16384 < S ≤ 32768` 自动退化 dense；`S > 32768` 触发 §3 的 OOM 链。

**v2 出路**：与 §3 同——CANN 升级 / Triton-Ascend / dense 分块。

**状态**：v1 设计代价，behavior 正常，记录于报告"长上下文能力"小节。

---

## 10. accelerate `device_map="auto"` 按层切，**不切层内 attention**

**触发**：M5 128K 4 卡 `device_map=auto` 时，`npu-smi info` 显示 NPU0 = 58 GiB（满）、NPU1-3 各 ~4 GiB（仅权重副本）。

**根因**：accelerate 在 **layer 边界**切模型；attention forward 的中间激活（QK^T、attn_weights、bool mask、output）都必须在该层所在的那张卡上分配。多卡只摊薄了**模型权重**，对**单层 attention 的瞬时显存峰值毫无帮助**。

**结论**：要把超长 ctx 跑起来，必须解决**单卡内 attention 计算的显存峰值**（即 §3 的 bool mask + KV cache + attn 激活），不能指望靠多卡。

**v2 出路**：序列并行（SP）或张量并行（TP）。v1 明确不做。

**状态**：v1 设计代价；记录于报告"长上下文能力"小节。

---

## 11. example 脚本本地权重路径与 best_pattern key 解耦（已修）

**触发**：M5 实机用户的 Llama-3.1-8B 权重在本地路径 `/data/.../Meta-Llama-3.1-8B-Instruct`，但 `examples/run_hf_minimal.py` 原始版本把 `--model` 同时用于 `from_pretrained()` 和 `MInference(model_name=...)`；本地路径不在 `MODEL2PATH` 里，patch 查 best_pattern 失败。

**解法**：加 `--model-path` 可选参数：`from_pretrained()` 走 `--model-path or --model`，`MInference(model_name=...)` 仍用 `--model`（必须是 MODEL2PATH 的 key）。

**修复位置**：`MInference-NPU/examples/run_hf_minimal.py` argparse 段 + load 段。

**状态**：已修。

---

## 12. 输出 `torch_dtype` deprecation warning（环境侧，无害）

**触发**：每次跑 HF 例子都会有

```
`torch_dtype` is deprecated! Use `dtype` instead!
```

**根因**：transformers 4.57 改名 `torch_dtype` → `dtype`，旧 kwargs 仍向后兼容但 warn。`run_hf_minimal.py` 当前用的是 `torch_dtype=torch.float16`。

**解法**：未改（向后兼容期内无影响）；v1 v2 收尾时可一并改成 `dtype=`。

**状态**：低优先，记录待清理。

---

## 13. Ascend toolkit owner 警告（环境侧，无害）

**触发**：每次 import torch_npu 都会有

```
UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/8.1.RC1 owner does not match the current owner.
```

**根因**：CANN 安装时使用 root，当前 conda 用户读不到属主，torch_npu 的 `collect_env` 校验 owner 不一致。

**解法**：不影响功能，可加 `PYTHONWARNINGS=ignore::UserWarning:torch_npu.utils.collect_env` 过滤；或运维侧 chown。

**状态**：环境侧，未处理。

---

## 14. v1 总结：踩坑分类

| 类别 | 数量 | 代表条目 |
|---|---|---|
| 已在 v1 修复 | 5 | §2、§4、§5、§11、§6（约定） |
| v1 设计代价（v2 解决） | 4 | §3、§8、§9、§10 |
| 环境侧（外部依赖） | 4 | §1、§7（认知）、§12、§13 |

**v2 必做的优先级**（按重要性）：

1. **§8 消除 per-head Python 循环** —— 这是 v1 最大性能瓶颈，修了之后 minference 才有真正的加速空间
2. **§3 + §9 解决 O(S²) bool mask** —— 才能把 ctx 上限推到 128K+
3. **§10 序列并行 / 张量并行** —— 把多卡显存真正用起来
4. **§1 升级 CANN → 8.2+** —— 顺带 §2 的 sparse_mode 修复可能省 mask 显存
