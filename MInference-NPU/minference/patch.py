# Copyright (c) 2024-2025 Microsoft
# Copyright (c) 2026 (NPU 适配)
# Licensed under The MIT License [see LICENSE for details]
"""HF transformers monkey patch for the PR-4 NPU path.

Only the attention modules are patched.  The current target is Phi-3 style
causal LM inference on Ascend NPU with either dense baseline or best-pattern
grouped sparse attention.
"""

from __future__ import annotations

from typing import Any

import torch

from .modules.minference_forward import (
    gather_last_q_sparse_topk,
    init_minference_parameters,
    minference_forward,
)

__all__ = [
    "minference_patch",
    "patch_hf",
    # Compatibility stubs: not part of the trimmed current scope.
    "minference_patch_kv_cache_cpu",
    "minference_patch_with_kvcompress",
    "minference_patch_vllm",
    "new_patch",
]


# ----------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------


def _patch_causal_mask_helper(attention_module: torch.nn.Module) -> None:
    """Skip Phi/Llama-family 4D causal mask materialization when it is unused.

    The patched attention forward in this workspace ignores HF's prebuilt
    4D mask and applies causal semantics inside each backend kernel.  Phi-3's
    model forward still calls ``_prepare_4d_causal_attention_mask`` before the
    layer loop, which costs O(S^2) time and memory.  For the common inference
    case with no padding (2D attention_mask is all ones), returning ``None`` is
    equivalent for our patched attention modules.

    If a real padding mask is detected, fall back to HF's original helper.
    """

    import importlib

    module = importlib.import_module(attention_module.__class__.__module__)
    helper_name = "_prepare_4d_causal_attention_mask"
    original = getattr(module, helper_name, None)
    if original is None or getattr(original, "_minference_skip4d", False):
        return

    def _minference_prepare_4d_causal_attention_mask(
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
        sliding_window=None,
    ):
        if attention_mask is None:
            return None
        try:
            # In our generation probes the mask is [B, S] and all ones.  The
            # small reduction is cheap and avoids materializing [B, 1, S, S].
            has_padding = bool((attention_mask == 0).any().item())
        except Exception:  # noqa: BLE001
            has_padding = True
        if not has_padding:
            return None
        return original(
            attention_mask,
            input_shape,
            inputs_embeds,
            past_key_values_length,
            sliding_window=sliding_window,
        )

    _minference_prepare_4d_causal_attention_mask._minference_skip4d = True  # type: ignore[attr-defined]
    _minference_prepare_4d_causal_attention_mask._minference_original = original  # type: ignore[attr-defined]
    setattr(module, helper_name, _minference_prepare_4d_causal_attention_mask)


def minference_patch(model: torch.nn.Module, config: Any) -> torch.nn.Module:
    """把 `model.model.layers[*].self_attn.forward` 替换为 NPU 版 minference_forward。

    Args:
        model:  HF AutoModelForCausalLM 实例（当前主要验证 Phi-3）
        config: MInferenceConfig（仅读取 `attn_kwargs` / `starting_layer` / `config_path`）

    Returns:
        被 patch 过的 model（原地修改，返回值是 convenience）
    """
    # Detect the attention class and patch all layers of that class.
    try:
        attention_module = model.model.layers[0].self_attn
    except AttributeError as e:
        raise RuntimeError(
            "model.model.layers[0].self_attn 不存在。当前只支持 Llama-family/Phi-style 结构。"
        ) from e

    AttentionClass = attention_module.__class__
    forward_closure = minference_forward()
    _patch_causal_mask_helper(attention_module)

    # Inject runtime settings so each attention layer can initialize itself.
    model.config.starting_layer = config.starting_layer
    model.config.config_path = config.config_path
    model.config.minference_attn_type = config.attn_type

    def update_module(m: torch.nn.Module) -> None:
        if isinstance(m, AttentionClass):
            m.init_minference_parameters = init_minference_parameters.__get__(
                m, AttentionClass
            )
            m.gather_last_q_sparse_topk = gather_last_q_sparse_topk.__get__(
                m, AttentionClass
            )
            m.forward = forward_closure.__get__(m, AttentionClass)

    model.apply(update_module)
    print(
        f"[MInference-NPU] Patched {AttentionClass.__name__}.forward "
        f"(starting_layer={config.starting_layer}, attn_type={config.attn_type})"
    )
    return model


def patch_hf(
    model: torch.nn.Module,
    attn_type: str = "minference",
    attn_kwargs: dict | None = None,
    **kwargs: Any,
) -> torch.nn.Module:
    """Compatibility entry matching upstream ``patch_hf``."""
    attn_kwargs = dict(attn_kwargs or {})
    attn_kwargs.update(kwargs)

    if attn_type not in ("minference", "dense"):
        raise NotImplementedError(
            f"patch_hf attn_type={attn_type!r} is outside the trimmed PR-4 scope. "
            "Use 'minference' for path-B sparse probes or 'dense' for the baseline."
        )

    from .minference_configuration import MInferenceConfig

    config = MInferenceConfig(
        attn_type=attn_type,
        attn_kwargs=attn_kwargs,
        starting_layer=attn_kwargs.get("starting_layer", -1),
        config_path=attn_kwargs.get("config_path"),
    )
    return minference_patch(model, config)


# ----------------------------------------------------------------------------
# Unsupported upstream compatibility names.
# ----------------------------------------------------------------------------


def minference_patch_kv_cache_cpu(model):
    raise NotImplementedError(
        "KV cache CPU offload is outside the trimmed PR-4 path-B scope."
    )


def minference_patch_with_kvcompress(model, config):
    raise NotImplementedError(
        "KV compression integrations are outside the trimmed PR-4 path-B scope."
    )


def minference_patch_vllm(*args, **kwargs):
    raise NotImplementedError(
        "vLLM integration is outside the trimmed PR-4 path-B scope."
    )


def new_patch(model, config):
    """Upstream ``new_patch`` dispatches features removed from this workspace."""
    raise NotImplementedError(
        f"new_patch (attn_type={config.attn_type!r}, kv_type={config.kv_type!r}) "
        "is outside the trimmed PR-4 path-B scope."
    )
