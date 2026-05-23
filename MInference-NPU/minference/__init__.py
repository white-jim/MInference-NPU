# Copyright (c) 2024 Microsoft
# Copyright (c) 2026 (NPU 适配)
# Licensed under The MIT License [see LICENSE for details]
"""MInference-NPU — Ascend NPU 适配版（v1）

与上游 MInference 1.0 的差异：
- v1 不构建 minference.cuda 扩展；索引展开由 backend_npu.cuda_shim 提供占位（M4 实现）
- v1 三种稀疏分支（vertical_and_slash / block_sparse / stream_llm）在 M1 阶段全部退化为
  backend_npu.dense_attention；M2/M3/M4 逐个替换回真稀疏 kernel
- v1 仅支持 HF transformers + torch_npu 宿主；vLLM 集成 / KV 压缩 / 分布式（dist_ops）/
  dilated/static/tri_shape/tri_mix/inf_llm/flexprefill/xattention 都是 v2 计划项
"""

from .configs.model2path import get_support_models
from .minference_configuration import MInferenceConfig
from .models_patch import MInference
from .patch import minference_patch, patch_hf
from .version import VERSION as __version__

# v1 阶段稀疏算子接口的"门面"：M2-M4 完成时会指向真 NPU 稀疏 kernel；M1 阶段先指向
# dense fallback。三者的签名与上游对齐（多余参数被忽略），方便上游测试代码原样跑通。
from .backend_npu import dense_attention as _dense


def vertical_slash_sparse_attention(q, k, v, vertical_topk=None, slash=None):  # noqa: D401
    """v1 M1: 等价于 backend_npu.dense_attention(q, k, v, causal=True)。M4-b 时替换。"""
    return _dense(q, k, v, causal=True)


def block_sparse_attention(q, k, v, topk=None):  # noqa: D401
    """v1 M1: 等价于 backend_npu.dense_attention(q, k, v, causal=True)。M3 时替换。"""
    return _dense(q, k, v, causal=True)


def streaming_forward(q, k, v, n_init=None, n_local=None):  # noqa: D401
    """v1 M1: 等价于 backend_npu.dense_attention(q, k, v, causal=True)。M2 时替换。"""
    return _dense(q, k, v, causal=True)

__all__ = [
    "MInference",
    "MInferenceConfig",
    "minference_patch",
    "patch_hf",
    "get_support_models",
    "vertical_slash_sparse_attention",
    "block_sparse_attention",
    "streaming_forward",
]
