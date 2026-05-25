# Copyright (c) 2024-2025 Microsoft
# Copyright (c) 2026 (NPU 适配)
# Licensed under The MIT License [see LICENSE for details]
"""HF transformers monkey-patch — NPU v1 简化版。

与上游 `patch_upstream.py` 的差异：
- 移除 vLLM 路径（`minference_patch_vllm*`、`new_patch` 的 prefill_forwards 分发等）
- 移除 KV cache CPU offload（`minference_patch_kv_cache_cpu`）—— NPU HBM 64GB 单卡够
  长上下文场景，且 NPU↔Host 拷贝代价远大于 GPU 的 PCIe；留到 v2 再视需求接入
- 移除 KV 压缩（`minference_patch_with_kvcompress`）—— v1 排除项
- 移除 inf_llm / chatglm 特殊路径 —— v1 范围 Llama / Qwen / Mistral 已够
- 不替换 `LlamaModel.forward` / `LlamaForCausalLM.forward` —— 上游替换是为了 bypass HF
  的 causal mask 构造 + 注入 flash_attn 特定逻辑；NPU 上 `npu_fusion_attention` 用
  `sparse_mode=2` 自带 causal，HF 默认 forward 就能正常工作

保留：
- `minference_patch(model, config)` —— 主入口，monkey-patch LlamaAttention.forward
- `patch_hf(model, attn_type, attn_kwargs)` —— 兼容上游 API 的轻封装

注意所有 flash_attn / sgl_kernel / vllm_flash_attn 的 import 都已 try-except 守护，
import 阶段不会因为这些包缺失而挂。
"""

from __future__ import annotations

from typing import Any

import torch

from .modules.minference_forward import (
    gather_last_q_vertical_slash_topk_v4,
    init_minference_parameters,
    minference_forward,
)

__all__ = [
    "minference_patch",
    "patch_hf",
    # 占位：以下接口在 v1 不实现，调用会 raise NotImplementedError，保留以兼容上游 API
    "minference_patch_kv_cache_cpu",
    "minference_patch_with_kvcompress",
    "minference_patch_vllm",
    "new_patch",
]


# ----------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------


def minference_patch(model: torch.nn.Module, config: Any) -> torch.nn.Module:
    """把 `model.model.layers[*].self_attn.forward` 替换为 NPU 版 minference_forward。

    Args:
        model:  HF AutoModelForCausalLM 实例（Llama / Qwen / Mistral）
        config: MInferenceConfig（仅读取 `attn_kwargs` / `starting_layer` / `config_path`）

    Returns:
        被 patch 过的 model（原地修改，返回值是 convenience）
    """
    # 探测 attention 类（不同模型族类名不一样：LlamaAttention / Qwen2Attention / MistralAttention）
    try:
        attention_module = model.model.layers[0].self_attn
    except AttributeError as e:
        raise RuntimeError(
            "model.model.layers[0].self_attn 不存在。当前只支持 Llama-family 模型结构。"
        ) from e

    AttentionClass = attention_module.__class__
    forward_closure = minference_forward()

    # 把 starting_layer / config_path / attn_type 注入到 model.config，以便每层 attention 的
    # init_minference_parameters() 能读到（与上游 models_patch_upstream.py:96 同步）。
    # attn_type 用于在 forward 中区分 dense（裸 npu_fusion_attention，绕开 per-head 调度）
    # 与 minference（按 best_pattern 走 per-head 调度）。
    model.config.starting_layer = config.starting_layer
    model.config.config_path = config.config_path
    model.config.minference_attn_type = config.attn_type

    def update_module(m: torch.nn.Module) -> None:
        if isinstance(m, AttentionClass):
            m.init_minference_parameters = init_minference_parameters.__get__(
                m, AttentionClass
            )
            m.gather_last_q_vertical_slash_topk_v4 = (
                gather_last_q_vertical_slash_topk_v4.__get__(m, AttentionClass)
            )
            m.forward = forward_closure.__get__(m, AttentionClass)

    model.apply(update_module)
    print(
        f"[MInference-NPU] Patched {AttentionClass.__name__}.forward "
        f"(starting_layer={config.starting_layer}, dense fallback for all sparse modes)"
    )
    return model


def patch_hf(
    model: torch.nn.Module,
    attn_type: str = "minference",
    attn_kwargs: dict | None = None,
    **kwargs: Any,
) -> torch.nn.Module:
    """与上游 `patch_hf` 同名的兼容入口。v1 仅支持 `attn_type="minference"`。

    其他 attn_type（a_shape / tri_shape / inf_llm / flexprefill / xattention 等）
    会 raise NotImplementedError —— 这些是 v1 排除项。
    """
    attn_kwargs = dict(attn_kwargs or {})
    attn_kwargs.update(kwargs)

    if attn_type not in ("minference", "dense"):
        raise NotImplementedError(
            f"patch_hf attn_type='{attn_type}' 在 v1 不支持。"
            " v1 仅支持 'minference'（dense fallback）与 'dense'（裸 npu_fusion_attention）。"
            " 其它 attn_type（a_shape/tri_shape/inf_llm/flexprefill/xattention）是 v1 排除项。"
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
# v1 不支持的 API（占位，保留命名以兼容上游 import）
# ----------------------------------------------------------------------------


def minference_patch_kv_cache_cpu(model):
    raise NotImplementedError(
        "KV cache CPU offload 在 v1 不支持。NPU 910B HBM 64GB 单卡足以承载常见长上下文场景。"
    )


def minference_patch_with_kvcompress(model, config):
    raise NotImplementedError(
        "KV 压缩（snapkv/pyramidkv/quest/kivi/retr_attn/leank/streamingllm）是 v1 排除项。"
    )


def minference_patch_vllm(*args, **kwargs):
    raise NotImplementedError(
        "vLLM 集成是 v1 排除项。v1 仅支持 HF transformers + torch_npu。"
    )


def new_patch(model, config):
    """上游 `new_patch` 用 prefill_forwards/decoding_forwards 派发到 flexprefill /
    xattention / inf_llm / leank / kivi 等 v1 排除项。v1 内不实现，直接回退到 minference_patch。"""
    raise NotImplementedError(
        f"new_patch (attn_type='{config.attn_type}', kv_type='{config.kv_type}') 是 v1 排除项。"
        " 请直接调用 minference_patch() 或将 attn_type 设为 'minference'。"
    )
