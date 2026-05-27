# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""Best-pattern config registry for the current PR-4 focus.

Only Phi-3 128K configs are kept in this trimmed workspace.  Use
``--config-path`` for explicit probe configs.
"""

from __future__ import annotations

import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL2PATH = {
    "microsoft/Phi-3-mini-128k-instruct": os.path.join(
        BASE_DIR, "Phi_3_mini_128k_instruct_pathb_stream_llm_aligned_dense_others.json"
    ),
}


def get_support_models():
    return list(MODEL2PATH.keys())


def check_path():
    for name, path in MODEL2PATH.items():
        assert os.path.exists(path), f"{name} config does not exist: {path}"


if __name__ == "__main__":
    check_path()
