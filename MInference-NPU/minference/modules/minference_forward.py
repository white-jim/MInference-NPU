# Copyright (c) 2024-2025 Microsoft
# Copyright (c) 2026 (NPU 适配)
# Licensed under The MIT License [see LICENSE for details]
"""MInference forward — NPU v1（dense fallback）版。

与上游 `minference_forward_upstream.py` 的差异：
- **三种稀疏分支（vertical_and_slash / stream_llm / block_sparse）暂时全部退化为
  `backend_npu.dense_attention`**。per-head 循环骨架保留不动，M2/M3/M4 时按分支逐个
  替换回真 NPU 稀疏 kernel。
- 移除 vllm / flexprefill / kvcompression / quest / snapkv / leank / tri_mix / xattention
  等 v1 排除项的 import 与代码路径。
- 移除 v1 排除的 attn_type 分支：dilated1/dilated2/static_pattern/vs_only/a_shape/tri_shape
  在本文件不出现（这些分支由 patch.py 的 attn_type 路由提前 raise / 走 dense）。
- 所有 `device="cuda"` 改为 device-agnostic：跟随输入 tensor 的 device，保证 accelerate
  自动多卡（device_map="auto"）切层后不挂。
- `q_len == 1` decode 路径走 `backend_npu.decode_dense`（包 npu_incre_flash_attention）。

保留：
- per-head 循环 + best_pattern 加载 + 在线估计代码骨架（M4-b 时复用）
- sum_all_diagonal_matrix 反对角线求和（M4-b 时复用）
- RotaryEmbeddingESM 类型探测 / get_cos_sin（patch.py 不再依赖 flash_attn 的 apply_rotary_emb）
"""

from __future__ import annotations

import inspect
import json
import math
import os
from importlib import import_module

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import rotate_half

from ..backend_npu import dense_attention, decode_dense
from ..ops.streaming_kernel_npu import streaming_forward as _streaming_forward
from ..ops.block_sparse_kernel_npu import block_sparse_attention as _block_sparse_attention
from ..ops.vertical_slash_kernel_npu import vertical_slash_sparse_attention as _vs_sparse_attention

__all__ = [
    "init_minference_parameters",
    "gather_last_q_vertical_slash_topk_v4",
    "minference_forward",
    "sum_all_diagonal_matrix",
    "set_rope_type",
    "get_cos_sin",
    "apply_rotary_pos_emb_single",
]


# ----------------------------------------------------------------------------
# 全局状态（device-agnostic 懒初始化）
# ----------------------------------------------------------------------------

LAST_Q = 64
_LAST_Q_MASK_CACHE: dict[torch.device, torch.Tensor] = {}
ROPE_TYPE: str | None = None


def _last_q_mask(device: torch.device) -> torch.Tensor:
    """`[1, 1, LAST_Q, LAST_Q]` 因果 mask，按 device 缓存。"""
    mask = _LAST_Q_MASK_CACHE.get(device)
    if mask is None:
        arange = torch.arange(LAST_Q, device=device)
        mask = arange[None, None, :, None] >= arange[None, None, None, :]
        _LAST_Q_MASK_CACHE[device] = mask
    return mask


# ----------------------------------------------------------------------------
# RoPE 类型探测（与上游同源，逻辑不变）
# ----------------------------------------------------------------------------


def set_rope_type(self) -> None:
    """探测 self.rotary_emb 的接口签名，缓存到 ROPE_TYPE。HF 各版本签名不一致。"""
    global ROPE_TYPE
    if ROPE_TYPE is not None:
        return
    sig = inspect.signature(self.rotary_emb.forward).parameters
    if "seq_len" in sig:
        if "position_ids" in sig:
            ROPE_TYPE = "seq_len,position_ids"
        else:
            ROPE_TYPE = "seq_len"
    elif "max_seq_len" in sig:
        ROPE_TYPE = "max_seq_len"
    else:
        ROPE_TYPE = "position_ids"


def get_cos_sin(self, value_states, kv_seq_len, position_ids):
    # device-agnostic：与上游同源，但去掉了对 self.rotary_emb.inv_freq 的依赖（有些 RoPE
    # 实现没这个 buffer），改为统一对齐到 value_states.device
    if position_ids is not None and value_states.device != position_ids.device:
        position_ids = position_ids.to(value_states.device)

    if ROPE_TYPE == "seq_len":
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    elif ROPE_TYPE == "seq_len,position_ids":
        cos, sin = self.rotary_emb(value_states, position_ids=position_ids, seq_len=kv_seq_len)
    elif ROPE_TYPE == "max_seq_len":
        if position_ids is not None and position_ids[0][0] < 0:
            kv_seq_len -= position_ids[0][0].item()
            position_ids = position_ids - position_ids[0][0]
        cos = self.rotary_emb(kv_seq_len)
        if position_ids is not None:
            cos = cos[position_ids.to(cos.device)]
        else:
            cos = cos[None, :kv_seq_len]
        sin = None
    else:
        cos, sin = self.rotary_emb(value_states, position_ids)
    return cos, sin


def apply_rotary_pos_emb_single(q, cos, sin, position_ids, unsqueeze_dim: int = 1):
    if cos.dim() == 2:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    else:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin)


# ----------------------------------------------------------------------------
# Per-layer 参数初始化（best_pattern 加载）
# ----------------------------------------------------------------------------


def init_minference_parameters(self) -> None:
    if getattr(self, '_minference_initialized', False):
        return
    config = self.config.to_dict()
    self.starting_layer = config.get("starting_layer", 0)
    self.is_search = config.get("is_search", False)
    self.ne_inf = None
    self.config_path = config.get("config_path", "")
    # attn_type 由 patch.py 注入到 model.config.minference_attn_type；缺省按 minference 处理
    # （兼容直接构造 MInferenceConfig 但忘了 patch 的边角调用）。
    self._minference_attn_type = config.get("minference_attn_type", "minference")

    if self.config_path and os.path.exists(self.config_path):
        with open(self.config_path) as f:
            all_patterns = json.load(f)
        if self.layer_idx < len(all_patterns):
            self.best_pattern = {
                int(ii): jj
                for ii, jj in all_patterns[self.layer_idx].items()
            }
        else:
            self.best_pattern = {}
    else:
        self.best_pattern = {}
    self.vertical, self.slash = None, None

    if "apply_rotary_pos_emb" not in self.__dict__:
        global apply_rotary_pos_emb
        # 新版 transformers 把 rotary_emb 迁到 model level，attention module 已无该属性。
        # 此时回退到 attention 自身的 module path 找 apply_rotary_pos_emb（同包内可见）。
        if hasattr(self, "rotary_emb"):
            model_path = self.rotary_emb.__class__.__module__
        else:
            model_path = self.__class__.__module__
        apply_rotary_pos_emb = getattr(
            import_module(model_path), "apply_rotary_pos_emb"
        )
        self.apply_rotary_pos_emb = True
    self._minference_initialized = True


# ----------------------------------------------------------------------------
# 算法工具（M4-b 时复用 sum_all_diagonal_matrix）
# ----------------------------------------------------------------------------


def sum_all_diagonal_matrix(mat: torch.Tensor) -> torch.Tensor:
    """反对角线求和：把 `[B, H, n, m]` 矩阵的每一条反对角线之和拼成 `[B, H, n, n+m-1]`。

    实现技巧：把 mat 左右各 pad n 个 0，再用 as_strided 把 (2n+m, 2n+m+1) 的 stride
    映射出每条反对角线，最后 sum dim=2。device-agnostic。M4-b vertical 估计时复用。
    """
    b, h, n, m = mat.shape
    zero_mat = torch.zeros((b, h, n, n), device=mat.device, dtype=mat.dtype)
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)
    mat_strided = mat_padded.as_strided(
        (b, h, n, n + m), (h * n * (2 * n + m), n * (2 * n + m), 2 * n + m + 1, 1)
    )
    sum_diags = torch.sum(mat_strided, 2)
    return sum_diags[:, :, 1:]


def gather_qkv(q, k, v, attention_mask):
    """PyTorch eager dense attention。在没有 flash_attn / npu_fusion_attention 时兜底。"""
    attn_weights = (
        torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.size(-1)) + attention_mask
    )
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        q.dtype
    )
    return torch.matmul(attn_weights, v)


# ----------------------------------------------------------------------------
# Per-head 调度器（M1：三种稀疏全部 dense；M2-M4 逐个替换）
# ----------------------------------------------------------------------------


def _vertical_and_slash_kernel(self, q, k, v, vertical_size, slash_size):
    """在线估计 vertical/slash 索引，调用 vertical-slash 稀疏 attention。

    与上游 ``minference_forward_upstream.py:381 vertical_and_slash_kernel`` 等价：
    - 取最后 last_q 个 query 做一次 QK 估计。
    - 列和 → topk → v_idx（升序）。
    - 反对角线和 → topk → s_idx（降序，= q_len - 1 - diag_from_upper_right）。
    - 强制保留前 30 列（sink）和最近 100 条 slash（local window）。

    v2 PR-1：``vertical_size`` / ``slash_size`` 同时接受标量与长度为 H 的 list/tuple/tensor。
    传 list 时，每个 head 用各自的尺寸做 topk —— 通过先按 ``max(sizes)`` topk、再把
    超出本 head 配额的位置替换为 ``[..., 0:1]`` 的重复值（shim 与 mask 构造都对重复索引
    幂等），保证与 per-head 循环 bit-identical。
    """
    B, H, q_len, _ = q.shape
    last_q = min(LAST_Q, q_len)
    device = q.device

    # ---- 归一化尺寸为长度 H 的 Python list（per-head clamp） ----
    def _norm(sz, lower):
        if isinstance(sz, (list, tuple)):
            assert len(sz) == H, f"per-head size 长度 {len(sz)} != H {H}"
            return [min(q_len, max(int(x), lower)) for x in sz]
        if torch.is_tensor(sz):
            assert sz.numel() == H, f"per-head size tensor numel {sz.numel()} != H {H}"
            return [min(q_len, max(int(x), lower)) for x in sz.tolist()]
        return [min(q_len, max(int(sz), lower))] * H

    vs_per_h = _norm(vertical_size, 30)
    ss_per_h = _norm(slash_size, 50)
    V_max = max(vs_per_h)
    S_max = max(ss_per_h)

    # QK 估计（仅用最后 last_q 个 query）
    qk = torch.matmul(q[:, :, -last_q:, :], k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    # 因果 mask（[1,1,last_q,last_q] 的右下角）
    lq_mask = _last_q_mask(device)[:, :, -last_q:, -last_q:]  # [1,1,last_q,last_q]
    qk[:, :, :, -last_q:] = torch.where(lq_mask, qk[:, :, :, -last_q:], float("-inf"))
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)  # [B,H,last_q,q_len]

    # Vertical：列和 → topk（保留前 30 sink 列）
    vertical = qk.sum(-2, keepdim=True)              # [B,H,1,q_len]
    vertical[..., :30] = torch.inf                   # 强制保留 sink
    v_idx = torch.topk(vertical, V_max, dim=-1).indices  # [B,H,1,V_max]
    v_idx = v_idx.reshape(B, H, V_max)               # [B,H,V_max]

    # per-head trim：head h 只想要前 vs_per_h[h] 项，余下位置改写为 v_idx[..., 0:1]
    # （shim 的 col_buf 对重复列幂等，mask scatter True 也幂等 → bit-identical）
    if any(x != V_max for x in vs_per_h):
        vs_t = torch.tensor(vs_per_h, device=device).view(1, H, 1)
        arange_v = torch.arange(V_max, device=device).view(1, 1, V_max)
        valid_v = arange_v < vs_t                    # [1,H,V_max]
        v_idx = torch.where(valid_v, v_idx, v_idx[..., 0:1])

    v_idx = v_idx.sort(dim=-1, descending=False).values.to(torch.int32)  # 升序

    # Slash：反对角线和 → topk（保留最近 100 条 local）
    slash_diag = sum_all_diagonal_matrix(qk)[..., :-last_q + 1]  # [B,H,q_len]
    slash_diag[..., -100:] = torch.inf                            # 强制保留 local
    s_raw_idx = torch.topk(slash_diag, S_max, dim=-1).indices    # [B,H,S_max]
    s_idx = (q_len - 1) - s_raw_idx                               # 上游约定

    # per-head trim：与 vertical 同样思路。slash 重复值在 _process_block 的 else
    # 分支（new_range_end <= range_end）会被跳过，不影响 block_offset。
    if any(x != S_max for x in ss_per_h):
        ss_t = torch.tensor(ss_per_h, device=device).view(1, H, 1)
        arange_s = torch.arange(S_max, device=device).view(1, 1, S_max)
        valid_s = arange_s < ss_t                    # [1,H,S_max]
        s_idx = torch.where(valid_s, s_idx, s_idx[..., 0:1])

    s_idx = s_idx.sort(dim=-1, descending=True).values.to(torch.int32)  # 降序

    return _vs_sparse_attention(q, k, v, v_idx, s_idx)


def gather_last_q_vertical_slash_topk_v4(self, q, k, v, head_id: int):
    """单 head 调度。

    M2: stream_llm 分支接入真 NPU streaming kernel。
    M3: block_sparse 分支接入真 NPU block-sparse kernel。
    M4: vertical_and_slash 分支接入真 NPU vertical-slash kernel。
    """
    q_len = q.shape[2]

    # decode 短路（与上游一致）
    if q_len == 1:
        return decode_dense(q, k, v)

    # best_pattern 缺省 → vertical_and_slash 默认参数（与上游 line 474 一致）
    ty, vertical_size, slash_size, _ = self.best_pattern.get(
        head_id, ("vertical_and_slash", 1000, 6096, 1)
    )

    if ty == "stream_llm":
        # M2: 上游约定 vertical_size → n_init, slash_size → n_local
        return _streaming_forward(q, k, v, n_init=vertical_size, n_local=slash_size)

    if ty == "block_sparse":
        # M3: vertical_size 存储 top_k（保留的 key block 数）
        return _block_sparse_attention(q, k, v, topk_blocks=int(vertical_size))

    # M4: vertical_and_slash — 在线估计 v_idx/s_idx → sparse attention
    return _vertical_and_slash_kernel(self, q, k, v, vertical_size, slash_size)


# ----------------------------------------------------------------------------
# 顶层 attention forward（替换 LlamaAttention.forward）
# ----------------------------------------------------------------------------


def minference_forward():
    """返回一个可绑定到 LlamaAttention.forward 的闭包。

    与上游 `minference_forward_upstream.py:497` 的差异：
    - decode 与 prefill 都用 `backend_npu.{decode_dense,dense_attention}`，不依赖 flash_attn
    - is_search 路径保留（仍然只在 GPU 上跑搜索；NPU 上 is_search=True 会 warn）

    签名适配 transformers 4.57.3（LlamaDecoderLayer 用全 kwargs 调 self_attn）：
    - 所有参数带默认值；`past_key_values` 用新版复数命名，旧版 `past_key_value` 通过 kwargs 兼容
    - `output_attentions` 在新版已删除（保留默认以兼容旧版 kwargs 误传）
    - `position_embeddings` 提升为命名参数；新版 RoPE 必走此路径
    - 返回 2-tuple `(attn_output, attn_weights)`（新版 LlamaDecoderLayer 期望）
    - `num_heads` / `num_key_value_heads` 改从 `self.config` 读取（4.57+ 不再挂在 attention module 上）
    """

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        output_attentions=False,
        **kwargs,
    ):
        # 兼容旧版 transformers：past_key_value（单数）落进 kwargs
        if past_key_values is None:
            past_key_values = kwargs.pop("past_key_value", None)

        self.init_minference_parameters()
        self.ne_inf = torch.finfo(hidden_states.dtype).min

        bsz, q_len, _ = hidden_states.size()

        # 新版 transformers (>=4.45) 不再把 num_heads / num_key_value_heads 挂在 attention
        # module 上，统一从 config 取；同时兼容旧版（实例属性优先）
        num_heads = getattr(self, "num_heads", None) or self.config.num_attention_heads
        num_kv_heads = (
            getattr(self, "num_key_value_heads", None)
            or self.config.num_key_value_heads
        )

        # QKV proj —— 兼容 q_proj/k_proj/v_proj 拆分式 和 qkv_proj 融合式
        if "q_proj" in self.__dict__["_modules"]:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else:
            qkv = self.qkv_proj(hidden_states)
            query_pos = num_heads * self.head_dim
            kv_pos = query_pos // self.num_key_value_groups
            query_states, key_states, value_states = torch.split(
                qkv, [query_pos, kv_pos, kv_pos], -1
            )

        query_states = query_states.view(
            bsz, q_len, num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, num_kv_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_values is not None:
            if self.layer_idx is None:
                raise ValueError(
                    "Cache structure requires layer_idx since transformers v4.36."
                )
            # 新版 Cache 仅保证 get_seq_length；旧版还有 get_usable_length
            if hasattr(past_key_values, "get_usable_length"):
                kv_seq_len += past_key_values.get_usable_length(
                    kv_seq_len, self.layer_idx
                )
            elif hasattr(past_key_values, "get_seq_length"):
                kv_seq_len += past_key_values.get_seq_length(self.layer_idx)

        # 新版 transformers 把 (cos, sin) 直接通过 position_embeddings 传给 attention
        # forward；旧版仍走 self.rotary_emb。两种路径都要支持。
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            if position_ids is not None and position_ids.device != cos.device:
                position_ids = position_ids.to(cos.device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        else:
            set_rope_type(self)
            cos, sin = get_cos_sin(self, value_states, kv_seq_len, position_ids)
            if ROPE_TYPE == "max_seq_len":
                if cos.device != query_states.device:
                    cos = cos.to(query_states.device)
                query_states = apply_rotary_pos_emb(query_states, cos)
                key_states = apply_rotary_pos_emb(key_states, cos)
            else:
                if position_ids is not None and position_ids.device != cos.device:
                    position_ids = position_ids.to(cos.device)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids
                )

        if past_key_values is not None and hasattr(past_key_values, "update"):
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # GQA: 把 K/V 复制成与 Q 同 head 数（per-head 循环简化）
        from transformers.models.llama.modeling_llama import repeat_kv

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.is_search:
            # search 流程在 NPU 上不推荐跑（速度慢、且搜索结果 GPU/NPU 通用），但保留接口
            import warnings

            warnings.warn(
                "is_search=True on NPU is supported but slow. "
                "推荐在 GPU 上搜出 best_pattern JSON 后迁到 NPU。",
                stacklevel=2,
            )

        if q_len != 1:
            # attn_type == "dense"：所有 head/layer 走 backend_npu.dense_attention，
            # 不进 per-head 调度，也不构造任何稀疏索引。dense 模式即裸 npu_fusion_attention
            # 基线，用于精度对照。
            if self._minference_attn_type == "dense":
                output = dense_attention(
                    query_states, key_states, value_states, causal=True
                )
            elif self.layer_idx < self.starting_layer or self.is_search:
                # 起始层 / search 模式：整层走 dense（不走 per-head 循环）
                output = dense_attention(
                    query_states, key_states, value_states, causal=True
                )
            else:
                # v2 PR-1：按 best_pattern 类型分组，vertical_and_slash heads 一次性
                # batched 调用 _vertical_and_slash_kernel，消除 per-head Python 循环。
                # stream_llm / block_sparse 暂保持 per-head（占比 < 5%，PR-2 再 batch）。
                H = query_states.size(1)
                vs_heads: list[int] = []
                vs_v_sizes: list[int] = []
                vs_s_sizes: list[int] = []
                stream_calls: list[tuple[int, int, int]] = []
                block_calls: list[tuple[int, int]] = []
                for head in range(H):
                    ty, vsz, ssz, _ = self.best_pattern.get(
                        head, ("vertical_and_slash", 1000, 6096, 1)
                    )
                    if ty == "vertical_and_slash":
                        vs_heads.append(head)
                        vs_v_sizes.append(int(vsz))
                        vs_s_sizes.append(int(ssz))
                    elif ty == "stream_llm":
                        stream_calls.append((head, int(vsz), int(ssz)))
                    elif ty == "block_sparse":
                        block_calls.append((head, int(vsz)))
                    else:
                        raise ValueError(
                            f"未知 best_pattern 类型 {ty!r}（layer={self.layer_idx} head={head}）"
                        )

                output = torch.empty_like(query_states)

                # vertical_and_slash: batched（PR-1 主路径）
                if vs_heads:
                    if len(vs_heads) == H:
                        # 全部 head 都是 VS：跳过 index_select / index_copy_ 直接整张量
                        out_vs = _vertical_and_slash_kernel(
                            self, query_states, key_states, value_states,
                            vs_v_sizes, vs_s_sizes,
                        )
                        output = out_vs
                    else:
                        head_idx_t = torch.tensor(
                            vs_heads, device=query_states.device, dtype=torch.long
                        )
                        q_vs = query_states.index_select(1, head_idx_t)
                        k_vs = key_states.index_select(1, head_idx_t)
                        v_vs = value_states.index_select(1, head_idx_t)
                        out_vs = _vertical_and_slash_kernel(
                            self, q_vs, k_vs, v_vs, vs_v_sizes, vs_s_sizes,
                        )
                        output.index_copy_(1, head_idx_t, out_vs)

                # stream_llm / block_sparse: per-head（PR-2 再 batch）
                for head, n_init, n_local in stream_calls:
                    q = query_states[:, head : head + 1]
                    k = key_states[:, head : head + 1]
                    v = value_states[:, head : head + 1]
                    output[:, head : head + 1] = _streaming_forward(
                        q, k, v, n_init=n_init, n_local=n_local
                    )
                for head, topk in block_calls:
                    q = query_states[:, head : head + 1]
                    k = key_states[:, head : head + 1]
                    v = value_states[:, head : head + 1]
                    output[:, head : head + 1] = _block_sparse_attention(
                        q, k, v, topk_blocks=topk
                    )
        else:
            # decode 路径：一次性整 head 调用，不走 per-head 循环（与上游 line 584 一致）
            output = decode_dense(query_states, key_states, value_states)

        attn_output = output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        # 新版 LlamaDecoderLayer 期望 (attn_output, attn_weights) 二元组；
        # 旧版 (attn_output, attn_weights, past_key_value) 在新版 transformers 下不再使用。
        return attn_output, None

    return forward
