# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from .phi3 import PHI_ATTN_FUNCS, Phi3ForCausalLM
from .qwen2 import QWEN_ATTN_FUNCS, Qwen2ForCausalLM

SUPPORTED_MODEL_SERIRS = {"Phi-3", "Qwen2.5"}

MODEL_TO_ATTN_FUNC = {
    "Phi-3": PHI_ATTN_FUNCS,
    "Qwen2.5": QWEN_ATTN_FUNCS,
}


MODEL_ID_TO_MODEL_CLS = {
    "Phi-3": Phi3ForCausalLM,
    "Qwen2.5": Qwen2ForCausalLM,
}

MODEL_ID_TO_PREFIX = {
    "Phi-3": "Phi3",
    "Qwen2.5": "Qwen2",
}


def _get_model_series(model_id: str):
    for model_series in SUPPORTED_MODEL_SERIRS:
        if model_series in model_id:
            return model_series
    raise ValueError(f"Model series not found in {model_id}")


def get_model_attn_funcs(model_id: str):
    model_series = _get_model_series(model_id)
    return MODEL_TO_ATTN_FUNC[model_series]


def get_model_cls(model_id: str):
    model_series = _get_model_series(model_id)
    return MODEL_ID_TO_MODEL_CLS[model_series]


def get_model_prefix(model_id: str):
    model_series = _get_model_series(model_id)
    return MODEL_ID_TO_PREFIX[model_series]
