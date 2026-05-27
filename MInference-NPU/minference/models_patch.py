# Copyright (c) 2024-2025 Microsoft
# Copyright (c) 2026 (NPU 适配)
# Licensed under The MIT License [see LICENSE for details]
"""Top-level MInference-NPU patch entry.

This trimmed workspace is focused on PR-4 TileLang path-B for Phi-3:
``stream_llm`` and ``block_sparse`` grouped sparse attention on Ascend NPU.
Only ``attn_type in {"minference", "dense", "hf"}`` is kept.
"""

from __future__ import annotations

import json
import os

from .minference_configuration import MInferenceConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_SUPPORTED_ATTN = ("minference", "dense", "hf")
_SUPPORTED_KV = ("dense",)


class MInference:
    """Top-level patch entry for the trimmed PR-4 workspace.

    用法：
        from minference import MInference
        model = AutoModelForCausalLM.from_pretrained(...).to("npu:0")
        model = MInference(
            attn_type="minference",
            model_name="microsoft/Phi-3-mini-128k-instruct",
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
        if attn_type not in _SUPPORTED_ATTN:
            raise ValueError(
                f"attn_type={attn_type!r} is outside the trimmed PR-4 scope. "
                f"Use one of {_SUPPORTED_ATTN}."
            )
        if kv_type not in _SUPPORTED_KV:
            raise ValueError(
                f"kv_type={kv_type!r} is outside the trimmed PR-4 scope. "
                f"Use one of {_SUPPORTED_KV}."
            )

        self.config = MInferenceConfig(
            attn_type=attn_type,
            model_name=model_name,
            config_path=config_path,
            starting_layer=starting_layer,
            kv_cache_cpu=False,
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

        # "dense" and "minference" share the same patch.  The forward path
        # selects dense baseline or best-pattern grouped scheduling.
        from .patch import minference_patch

        return minference_patch(model, self.config)
