# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""MInference-NPU trimmed PR-4 package."""

from .configs.model2path import get_support_models
from .version import VERSION as __version__

__all__ = [
    "MInference",
    "MInferenceConfig",
    "minference_patch",
    "patch_hf",
    "get_support_models",
    "block_sparse_attention",
    "streaming_forward",
]


def __getattr__(name):
    if name == "MInference":
        from .models_patch import MInference

        return MInference
    if name == "MInferenceConfig":
        from .minference_configuration import MInferenceConfig

        return MInferenceConfig
    if name == "minference_patch":
        from .patch import minference_patch

        return minference_patch
    if name == "patch_hf":
        from .patch import patch_hf

        return patch_hf
    if name == "block_sparse_attention":
        from .ops.block_sparse_kernel_npu import block_sparse_attention

        return block_sparse_attention
    if name == "streaming_forward":
        from .ops.streaming_kernel_npu import streaming_forward

        return streaming_forward
    raise AttributeError(name)
