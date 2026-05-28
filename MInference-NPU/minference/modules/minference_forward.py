# Copyright (c) 2024-2025 Microsoft
# Copyright (c) 2026 (NPU 适配)
# Licensed under The MIT License [see LICENSE for details]
"""Attention forward used by the trimmed PR-4 path-B workspace.

Only three per-head pattern types are active:

* ``dense``: dense baseline / non-target clean-probe heads.
* ``stream_llm``: Ascend hardware band + sink + LSE merge.
* ``block_sparse``: TileLang path-B block-sparse wrapper.

Legacy sparse patterns are intentionally not implemented in this trimmed tree.
Use the compact ``*_dense_others.json`` Phi-3 probe configs.
"""

from __future__ import annotations

import inspect
import json
import math
import os
from importlib import import_module

import torch
from transformers.models.llama.modeling_llama import rotate_half

from ..backend_npu import dense_attention, decode_dense
from ..ops.streaming_kernel_npu import streaming_forward as _streaming_forward
from ..ops.block_sparse_kernel_npu import block_sparse_attention as _block_sparse_attention
from ..ops.vertical_slash_kernel_npu import vertical_slash_sparse_attention as _vs_sparse_attention

__all__ = [
    "init_minference_parameters",
    "minference_forward",
    "set_rope_type",
    "sum_all_diagonal_matrix",
    "get_cos_sin",
    "apply_rotary_pos_emb_single",
    "gather_last_q_sparse_topk",
]


LAST_Q = 64
_LAST_Q_MASK_CACHE: dict[torch.device, torch.Tensor] = {}
ROPE_TYPE: str | None = None


def _block_sparse_head_chunk_size() -> int:
    raw = os.environ.get("MINFERENCE_BLOCK_SPARSE_HEAD_CHUNK")
    if raw is None or raw == "":
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _last_q_mask(device: torch.device) -> torch.Tensor:
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


def sum_all_diagonal_matrix(mat: torch.Tensor) -> torch.Tensor:
    """Sum anti-diagonals of ``[B,H,n,m]`` into ``[B,H,n+m-1]``.

    This is the upstream helper used by vertical-and-slash online index
    estimation.
    """
    b, h, n, m = mat.shape
    zero_mat = torch.zeros((b, h, n, n), device=mat.device, dtype=mat.dtype)
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)
    mat_strided = mat_padded.as_strided(
        (b, h, n, n + m), (h * n * (2 * n + m), n * (2 * n + m), 2 * n + m + 1, 1)
    )
    sum_diags = torch.sum(mat_strided, 2)
    return sum_diags[:, :, 1:]


def _vertical_and_slash_kernel(self, q, k, v, vertical_size, slash_size):
    """Estimate vertical/slash indices online and call vertical-slash attention."""
    B, H, q_len, _ = q.shape
    last_q = min(LAST_Q, q_len)
    device = q.device

    vertical_size = min(q_len, max(int(vertical_size), 30))
    slash_size = min(q_len, max(int(slash_size), 50))

    qk = torch.matmul(q[:, :, -last_q:, :], k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    lq_mask = _last_q_mask(device)[:, :, -last_q:, -last_q:]
    qk[:, :, :, -last_q:] = torch.where(lq_mask, qk[:, :, :, -last_q:], float("-inf"))
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)

    vertical = qk.sum(-2, keepdim=True)
    vertical[..., :30] = torch.inf
    v_idx = torch.topk(vertical, vertical_size, dim=-1).indices.reshape(B, H, vertical_size)
    v_idx = v_idx.sort(dim=-1, descending=False).values.to(torch.int32)

    slash_diag = sum_all_diagonal_matrix(qk)
    if last_q > 1:
        slash_diag = slash_diag[..., : -last_q + 1]
    slash_diag[..., -100:] = torch.inf
    s_raw_idx = torch.topk(slash_diag, slash_size, dim=-1).indices
    s_idx = (q_len - 1) - s_raw_idx
    s_idx = s_idx.sort(dim=-1, descending=True).values.to(torch.int32)

    return _vs_sparse_attention(q, k, v, v_idx, s_idx)


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
# Single-head compatibility dispatcher
# ----------------------------------------------------------------------------


def gather_last_q_sparse_topk(self, q, k, v, head_id: int):
    """Single-head dispatcher kept for compatibility with old call sites."""
    q_len = q.shape[2]

    # decode 短路（与上游一致）
    if q_len == 1:
        return decode_dense(q, k, v)

    ty, vertical_size, slash_size, _ = self.best_pattern.get(
        head_id, ("vertical_and_slash", 1000, 6096, 1)
    )

    if ty == "stream_llm":
        # best_pattern convention: vertical_size -> n_init, slash_size -> n_local
        return _streaming_forward(q, k, v, n_init=vertical_size, n_local=slash_size)

    if ty == "block_sparse":
        # best_pattern convention: vertical_size stores top-k key blocks.
        return _block_sparse_attention(q, k, v, topk_blocks=int(vertical_size))

    if ty == "dense":
        return dense_attention(q, k, v, causal=True)

    if ty == "vertical_and_slash":
        return _vertical_and_slash_kernel(self, q, k, v, vertical_size, slash_size)

    raise NotImplementedError(
        f"Unsupported pattern {ty!r}. This trimmed workspace only supports "
        "'dense', 'stream_llm', 'block_sparse', and 'vertical_and_slash'."
    )


# ----------------------------------------------------------------------------
# 顶层 attention forward（替换 LlamaAttention.forward）
# ----------------------------------------------------------------------------


def minference_forward():
    """返回一个可绑定到 LlamaAttention.forward 的闭包。

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
        legacy_attention_return = "past_key_value" in kwargs or self.__class__.__name__.startswith("Phi3")
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
                # Group heads by active pattern and parameters.
                H = query_states.size(1)
                dense_heads: list[int] = []
                vs_groups: dict[tuple[int, int], list[int]] = {}
                stream_groups: dict[tuple[int, int], list[int]] = {}
                block_groups: dict[int, list[int]] = {}
                for head in range(H):
                    ty, vsz, ssz, _ = self.best_pattern.get(
                        head, ("vertical_and_slash", 1000, 6096, 1)
                    )
                    if ty == "vertical_and_slash":
                        vs_groups.setdefault((int(vsz), int(ssz)), []).append(head)
                    elif ty == "stream_llm":
                        stream_groups.setdefault((int(vsz), int(ssz)), []).append(head)
                    elif ty == "block_sparse":
                        block_groups.setdefault(int(vsz), []).append(head)
                    elif ty == "dense":
                        dense_heads.append(head)
                    else:
                        raise ValueError(
                            f"未知 best_pattern 类型 {ty!r}（layer={self.layer_idx} head={head}）"
                        )

                output = torch.empty_like(query_states)

                for (vertical_size, slash_size), heads in vs_groups.items():
                    head_idx_t = torch.tensor(
                        heads, device=query_states.device, dtype=torch.long
                    )
                    q = query_states.index_select(1, head_idx_t)
                    k = key_states.index_select(1, head_idx_t)
                    v = value_states.index_select(1, head_idx_t)
                    out_vs = _vertical_and_slash_kernel(
                        self,
                        q,
                        k,
                        v,
                        vertical_size=vertical_size,
                        slash_size=slash_size,
                    )
                    output.index_copy_(1, head_idx_t, out_vs)

                if dense_heads:
                    if len(dense_heads) == H:
                        output = dense_attention(
                            query_states, key_states, value_states, causal=True
                        )
                    else:
                        head_idx_t = torch.tensor(
                            dense_heads, device=query_states.device, dtype=torch.long
                        )
                        q_dense = query_states.index_select(1, head_idx_t)
                        k_dense = key_states.index_select(1, head_idx_t)
                        v_dense = value_states.index_select(1, head_idx_t)
                        out_dense = dense_attention(q_dense, k_dense, v_dense, causal=True)
                        output.index_copy_(1, head_idx_t, out_dense)

                # stream_llm / block_sparse: same-parameter heads batched together.
                for (n_init, n_local), heads in stream_groups.items():
                    head_idx_t = torch.tensor(
                        heads, device=query_states.device, dtype=torch.long
                    )
                    q = query_states.index_select(1, head_idx_t)
                    k = key_states.index_select(1, head_idx_t)
                    v = value_states.index_select(1, head_idx_t)
                    out_stream = _streaming_forward(
                        q, k, v, n_init=n_init, n_local=n_local
                    )
                    output.index_copy_(1, head_idx_t, out_stream)

                for topk, heads in block_groups.items():
                    chunk_size = _block_sparse_head_chunk_size()
                    if chunk_size <= 0:
                        chunks = [heads]
                    else:
                        chunks = [
                            heads[start : start + chunk_size]
                            for start in range(0, len(heads), chunk_size)
                        ]
                    for chunk_heads in chunks:
                        head_idx_t = torch.tensor(
                            chunk_heads, device=query_states.device, dtype=torch.long
                        )
                        q = query_states.index_select(1, head_idx_t)
                        k = key_states.index_select(1, head_idx_t)
                        v = value_states.index_select(1, head_idx_t)
                        out_block = _block_sparse_attention(
                            q, k, v, topk_blocks=topk
                        )
                        output.index_copy_(1, head_idx_t, out_block)
        else:
            # decode 路径：一次性整 head 调用，不走 per-head 循环（与上游 line 584 一致）
            output = decode_dense(query_states, key_states, value_states)

        attn_output = output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        # 新版 LlamaDecoderLayer 期望 (attn_output, attn_weights) 二元组；Phi3 remote
        # modeling 仍无条件解包三元组，需保持旧 attention 返回协议。
        if legacy_attention_return:
            return attn_output, None, past_key_values
        return attn_output, None

    return forward
