# Copyright (c) 2024-2025 Microsoft
# Copyright (c) 2026 (NPU 适配)
# Licensed under The MIT License [see LICENSE for details]
"""MInference 顶层入口 —— NPU v1 简化版。

与上游 `models_patch_upstream.py` 的差异：
- 移除全部 v1 排除项的分支（dilated1/dilated2/static/a_shape/tri_shape/tri_mix/
  inf_llm/flexprefill/xattention/vllm_*/streaming2 等）
- 移除 KV-type 特殊参数（snapkv/pyramidkv/quest/kivi/leank/retr_attn/streamingllm 等）
- v1 仅保留 `attn_type ∈ {"minference", "dense", "hf"}`、`kv_type ∈ {"dense"}`

`MInference(...)` 实例可直接 call(model) —— 与上游 API 对称，便于把已有 HF 例子迁过来。
"""

from __future__ import annotations

import json
import os

from .minference_configuration import MInferenceConfig
from .patch import minference_patch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_V1_SUPPORTED_ATTN = ("minference", "dense", "hf")
_V1_SUPPORTED_KV = ("dense",)


class MInference:
    """v1 顶层入口。

    用法：
        from minference import MInference
        model = AutoModelForCausalLM.from_pretrained(...).to("npu:0")
        model = MInference(
            attn_type="minference",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
        )(model)
    """

    def __init__(
        self,
        attn_type: str = "minference",
        model_name: str | None = None,
        config_path: str | None = None,
        starting_layer: int = -1,
        kv_type: str = "dense",
        is_search: bool = False,
        attn_kwargs: dict | None = None,
        **kwargs,
    ):
        if attn_type not in _V1_SUPPORTED_ATTN:
            raise ValueError(
                f"attn_type='{attn_type}' is v1 排除项。v1 仅支持 {_V1_SUPPORTED_ATTN}。"
            )
        if kv_type not in _V1_SUPPORTED_KV:
            raise ValueError(
                f"kv_type='{kv_type}' is v1 排除项。v1 仅支持 {_V1_SUPPORTED_KV}（"
                "snapkv/pyramidkv/quest/kivi/retr_attn/leank/streamingllm 都在 v2 计划）。"
            )

        self.config = MInferenceConfig(
            attn_type=attn_type,
            model_name=model_name,
            config_path=config_path,
            starting_layer=starting_layer,
            kv_cache_cpu=False,  # v1 不支持 CPU offload
            kv_type=kv_type,
            is_search=is_search,
            attn_kwargs=attn_kwargs or {},
            **kwargs,
        )

    def __call__(self, model):
        return self.patch_model(model)

    def patch_model(self, model):
        if self.config.attn_type == "hf":
            # 显式不 patch，跑原生 HF eager attention（用于精度对照）
            return model

        if self.config.attn_type == "minference":
            if not self.config.is_search:
                if not self.config.config_path:
                    raise ValueError(
                        "attn_type='minference' 需要提供 model_name (查 MODEL2PATH) "
                        "或 config_path（指向 best_pattern JSON）"
                    )
                with open(self.config.config_path, "r") as f:
                    self.config.attn_kwargs.setdefault("best_pattern", json.load(f))

        # attn_type == "dense" 与 "minference" 共用 minference_patch；区别仅在于：
        # - "minference" 会按 best_pattern 走 per-head 调度（M1 阶段全部 dense fallback）
        # - "dense" 没有 best_pattern，所有 head 走 dense（结果等价，但少一遍 JSON 加载）
        return minference_patch(model, self.config)
