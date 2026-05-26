# Copyright (c) 2026 (NPU 适配 — PR-4-tl-sfa 集成精度闸门)
# Licensed under The MIT License [see LICENSE for details]
"""tilelang ``sparse_attention_fwd`` × 我们的 ``tilelang_indices.py`` 联调精度测试。

这是 PR-4-tl-BS / PR-4-tl-SL 之前的最后一道闸门：
    1. 用 ``block_indices_to_tilelang`` / ``stream_llm_to_tilelang`` 构造 Indices
    2. 喂给 tilelang ``sparse_attention_fwd`` 跑 NPU 上的稀疏 FA
    3. 与 PyTorch dense fp32 + 同样 Indices mask 出来的 reference 对比

如果这一步精度过了，说明我们的 Indices 语义与 tilelang kernel 的约定一致，
后续直接把 ``block_sparse_kernel_npu.py`` / ``streaming_kernel_npu.py`` 切换到这条路径即可。

运行（NPU 机器，flexhead-tl conda env）::

    python tests/test_tilelang_sfa_integration.py            # 跑全部
    python tests/test_tilelang_sfa_integration.py --probe    # 仅 probe API 签名（先做这步）
    python tests/test_tilelang_sfa_integration.py --case bs  # 仅 block-sparse case
    python tests/test_tilelang_sfa_integration.py --case sl  # 仅 stream_llm case

设计原则：
    * 非 pytest（``__test__ = False``）—— 这是 NPU 手跑脚本，pytest 会把它当 collection 失败
    * 详尽 stdout：每一步打印 shape / dtype / max_abs_diff，便于失败时定位
    * API probe 在最前：tilelang ``sparse_attention_fwd`` 可能在不同 module 路径下，
      或返回 compiled function vs 接受 Q/K/V/Indices 直接调用，我们先试探再用
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import math
import os
import sys
import traceback
from typing import Callable, Optional

# 自洽：把 MInference-NPU 仓库根（tests/ 的父目录）加进 sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

# 绕开 minference/__init__.py（它会 eager import transformers，
# flexhead-tl conda env 里没装）。直接按文件路径加载 tilelang_indices 子模块。
import importlib.util as _ilu

_TI_PATH = os.path.join(_REPO_ROOT, "minference", "ops", "tilelang_indices.py")
_ti_spec = _ilu.spec_from_file_location("tilelang_indices_standalone", _TI_PATH)
tilelang_indices = _ilu.module_from_spec(_ti_spec)
_ti_spec.loader.exec_module(tilelang_indices)
TILELANG_PAD_VALUE = tilelang_indices.TILELANG_PAD_VALUE
block_indices_to_tilelang = tilelang_indices.block_indices_to_tilelang
stream_llm_to_tilelang = tilelang_indices.stream_llm_to_tilelang

# 让 pytest 不要把它当测试模块（脚本里 max_abs_diff 不用 assert）
__test__ = False

# ---------------------------------------------------------------------------
# Import 探测：tilelang ``sparse_attention_fwd`` 可能在多个路径下
# ---------------------------------------------------------------------------

_CANDIDATE_IMPORT_PATHS = [
    # 最可能：直接从 examples 目录 import（用户源码安装时通常会把 examples 加进 sys.path）
    "tilelang.examples.sparse_flash_attention.example_sparse_flash_attn",
    "examples.sparse_flash_attention.example_sparse_flash_attn",
    # 备选：可能被收纳到 tilelang.ops / tilelang.kernels 之类
    "tilelang.ops.sparse_flash_attention",
    "tilelang.kernels.sparse_flash_attention",
    "tilelang.sparse_flash_attention",
]


def _probe_sparse_attention_fwd() -> tuple[Callable, str]:
    """从候选路径里找到 ``sparse_attention_fwd``，返回 (callable, source_module)."""
    errors = []
    for path in _CANDIDATE_IMPORT_PATHS:
        try:
            mod = importlib.import_module(path)
            if hasattr(mod, "sparse_attention_fwd"):
                return mod.sparse_attention_fwd, path
            errors.append(f"  {path}: module imported but has no sparse_attention_fwd")
        except ImportError as e:
            errors.append(f"  {path}: ImportError({e})")
    raise ImportError(
        "无法在任何候选路径找到 sparse_attention_fwd：\n"
        + "\n".join(errors)
        + "\n请检查 tilelang-ascend 源码安装时 examples 目录是否在 sys.path 里。"
        + " 若 examples 是源码 layout，可以 PYTHONPATH=<tilelang-ascend>/examples 重跑。"
    )


def _print_signature(func: Callable, source: str) -> None:
    print(f"[probe] sparse_attention_fwd from: {source}")
    print(f"[probe] type: {type(func).__name__}")
    try:
        sig = inspect.signature(func)
        print(f"[probe] signature: {func.__name__ if hasattr(func, '__name__') else 'sparse_attention_fwd'}{sig}")
    except (TypeError, ValueError) as e:
        print(f"[probe] inspect.signature failed: {e}")
    doc = inspect.getdoc(func)
    if doc:
        print("[probe] docstring (前 20 行):")
        for line in doc.splitlines()[:20]:
            print(f"    {line}")
    else:
        print("[probe] no docstring")


# ---------------------------------------------------------------------------
# PyTorch dense fp32 reference（按 Indices 做 sparse mask）
# ---------------------------------------------------------------------------


def _torch_sparse_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    is_causal: bool,
    pad_value: int,
) -> torch.Tensor:
    """fp32 dense + Indices mask 的黄金参考。

    Args:
        q, k, v: [B, H, S, D] fp16/bf16，BNSD layout
        indices: [B, S_q, H_kv, topk] int32，每个 (b, s_q, h_kv) 看到的 K token 位置
                 （pad_value 表示无效）
        is_causal: True 时上三角额外 mask
        pad_value: indices 里的 pad sentinel

    Returns:
        out: [B, H, S, D]，dtype 同 q
    """
    B, H, S_q, D = q.shape
    S_k = k.shape[2]
    H_kv = indices.shape[2]
    assert H_kv == H, f"v1 仅支持 MHA，H={H} kv_heads={H_kv}"
    topk = indices.shape[-1]

    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    scale = 1.0 / math.sqrt(D)

    # 构造 attention mask：[B, H, S_q, S_k] bool，True 表示 valid
    mask = torch.zeros(B, H, S_q, S_k, dtype=torch.bool, device=q.device)
    # indices: [B, S_q, H, topk] —— 散播到 mask
    valid_idx = indices != pad_value  # [B, S_q, H, topk]
    # clamp 防止 -1 索引出错，反正后面用 valid_idx mask
    safe_idx = indices.clamp(min=0, max=S_k - 1).to(torch.long)  # [B, S_q, H, topk]
    # 把每个 (b, s_q, h, k_pos) 写到 mask[b, h, s_q, k_pos]
    # 用 scatter_：mask.permute → [B, H, S_q, S_k]
    # 先把 idx + valid_idx permute 到 [B, H, S_q, topk]
    safe_idx_bhsq = safe_idx.permute(0, 2, 1, 3).contiguous()  # [B, H, S_q, topk]
    valid_bhsq = valid_idx.permute(0, 2, 1, 3).contiguous()
    # scatter True 到 mask
    mask.scatter_(3, safe_idx_bhsq, valid_bhsq)

    if is_causal:
        causal = torch.tril(torch.ones(S_q, S_k, dtype=torch.bool, device=q.device))
        mask = mask & causal

    qk = torch.matmul(q_f, k_f.transpose(-1, -2)) * scale  # [B, H, S_q, S_k]
    qk = qk.masked_fill(~mask, float("-inf"))
    # 全 -inf 的行（理论上 padding 行）softmax 出 NaN，置零防御
    all_masked = ~mask.any(dim=-1, keepdim=True)
    qk = qk.masked_fill(all_masked, 0.0)
    p = torch.softmax(qk, dim=-1)
    p = p.masked_fill(all_masked, 0.0)
    out = torch.matmul(p, v_f)
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# tilelang sparse_attention_fwd 调用适配器
# ---------------------------------------------------------------------------


def _build_kernel(
    sfa: Callable,
    heads: int,
    dim: int,
    topk: int,
    is_causal: bool,
    block_I: int,
) -> object:
    """sparse_attention_fwd 是 kernel 构造器，返回 compiled kernel。"""
    return sfa(
        heads=heads,
        dim=dim,
        tail_dim=dim,
        topk=topk,
        kv_stride=1,
        kv_group=1,
        sm_scale=1.0 / math.sqrt(dim),
        is_causal=is_causal,
        block_I=block_I,
    )


def _inspect_kernel(kernel: object) -> None:
    """打印 kernel 的关键属性，帮助确定调用顺序。"""
    print(f"[adapter] kernel type: {type(kernel).__name__}")
    interesting_attrs = ["params", "input_tensors", "buffer_map", "prim_func", "func"]
    for attr in interesting_attrs:
        if hasattr(kernel, attr):
            val = getattr(kernel, attr)
            print(f"[adapter] kernel.{attr} = {val!r}")
    # 如果有 torch_function，看它的 signature
    if hasattr(kernel, "torch_function"):
        try:
            sig = inspect.signature(kernel.torch_function)
            print(f"[adapter] kernel.torch_function signature: {sig}")
        except (TypeError, ValueError):
            pass


def _call_sparse_attention_fwd(
    sfa: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    is_causal: bool,
    block_I: int,
    arg_order: str = "qkvi",
) -> torch.Tensor:
    """构造 kernel 并按 ``arg_order`` 指定的顺序调用。

    arg_order 是 4 字符串，每个字符是 q/k/v/i 之一，例如 ``"qkvi"`` = ``kernel(q, k, v, indices)``。
    """
    B, H, _, D = q.shape
    topk = indices.shape[-1]
    kernel = _build_kernel(sfa, H, D, topk, is_causal, block_I)
    _inspect_kernel(kernel)

    tensor_map = {"q": q, "k": k, "v": v, "i": indices}
    args = [tensor_map[c] for c in arg_order]
    print(f"[adapter] calling kernel({', '.join(arg_order)}) with shapes "
          f"{[tuple(t.shape) for t in args]} dtypes {[str(t.dtype) for t in args]}")
    return kernel(*args)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def run_block_sparse_case(sfa: Callable, device: torch.device) -> bool:
    """Block-sparse case：随机选 K block，验证我们的 block_indices → Indices 与 kernel 对接。"""
    print("\n" + "=" * 60)
    print("[case] BLOCK-SPARSE")
    print("=" * 60)

    torch.manual_seed(0)
    B, H, S, D = 1, 4, 512, 128
    block_size_M = 64
    block_size_N = 64
    n_q_blocks = S // block_size_M
    n_k_blocks = S // block_size_N
    max_blocks = 4  # 每个 Q block 看 4 个 K block

    q = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, S, D, dtype=torch.float16, device=device)

    # 随机生成 block_indices [B, H, n_q_blocks, max_blocks]
    # 注意：因为 is_causal=True 由 kernel 自行处理，我们不必裁未来 block
    block_indices = torch.zeros(B, H, n_q_blocks, max_blocks, dtype=torch.int32, device=device)
    for b in range(B):
        for h in range(H):
            for q_blk in range(n_q_blocks):
                # 简单选 [0, 1, q_blk-1, q_blk]（保证有效，含 anchor + sliding）
                choices = [0, 1, max(0, q_blk - 1), q_blk]
                block_indices[b, h, q_blk] = torch.tensor(choices, dtype=torch.int32)

    indices = block_indices_to_tilelang(
        block_indices,
        S_q=S,
        block_size_M=block_size_M,
        block_size_N=block_size_N,
        kv_heads=H,
    ).to(device)
    print(f"[bs] indices shape={tuple(indices.shape)} dtype={indices.dtype}")
    print(f"[bs] topk={indices.shape[-1]}, block_I={block_size_N}, topk % block_I = {indices.shape[-1] % block_size_N}")

    # 跑 tilelang
    try:
        out_tl = _call_sparse_attention_fwd(
            sfa, q, k, v, indices, is_causal=True, block_I=block_size_N
        )
    except Exception:
        print("[bs] tilelang sparse_attention_fwd 调用失败：")
        traceback.print_exc()
        return False
    print(f"[bs] tilelang out shape={tuple(out_tl.shape)} dtype={out_tl.dtype}")

    # 跑 PyTorch ref
    out_ref = _torch_sparse_ref(q, k, v, indices.cpu(), is_causal=True, pad_value=TILELANG_PAD_VALUE).to(device)

    diff = (out_tl.float() - out_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"[bs] max_abs_diff={max_diff:.4e}  mean_abs_diff={mean_diff:.4e}")
    threshold = 5e-2
    ok = max_diff < threshold
    print(f"[bs] threshold={threshold:.0e}  result={'PASS' if ok else 'FAIL'}")
    return ok


def run_stream_llm_case(sfa: Callable, device: torch.device) -> bool:
    """Stream-LLM case：固定 anchor + sliding window，验证 stream_llm_to_tilelang 与 kernel 对接。"""
    print("\n" + "=" * 60)
    print("[case] STREAM-LLM (A-shape)")
    print("=" * 60)

    torch.manual_seed(1)
    B, H, S, D = 1, 4, 512, 128
    n_init, n_local = 64, 128

    q = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, S, D, dtype=torch.float16, device=device)

    indices = stream_llm_to_tilelang(
        B=B,
        S_q=S,
        kv_heads=H,
        n_init=n_init,
        n_local=n_local,
        block_size_N=64,
        device=device,
    )
    print(f"[sl] indices shape={tuple(indices.shape)} dtype={indices.dtype}")
    print(f"[sl] topk={indices.shape[-1]}, block_I=64, topk % block_I = {indices.shape[-1] % 64}")

    try:
        out_tl = _call_sparse_attention_fwd(
            sfa, q, k, v, indices, is_causal=True, block_I=64
        )
    except Exception:
        print("[sl] tilelang sparse_attention_fwd 调用失败：")
        traceback.print_exc()
        return False
    print(f"[sl] tilelang out shape={tuple(out_tl.shape)} dtype={out_tl.dtype}")

    out_ref = _torch_sparse_ref(q, k, v, indices.cpu(), is_causal=True, pad_value=TILELANG_PAD_VALUE).to(device)

    diff = (out_tl.float() - out_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"[sl] max_abs_diff={max_diff:.4e}  mean_abs_diff={mean_diff:.4e}")
    threshold = 5e-2
    ok = max_diff < threshold
    print(f"[sl] threshold={threshold:.0e}  result={'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="tilelang sparse_attention_fwd 集成精度测试")
    parser.add_argument("--probe", action="store_true", help="仅 probe API 签名后退出")
    parser.add_argument("--case", choices=["bs", "sl", "all"], default="all")
    args = parser.parse_args()

    # NPU 设备
    try:
        import torch_npu  # noqa: F401
        device = torch.device("npu:0")
        print(f"[env] torch_npu OK，device={device}")
    except ImportError as e:
        print(f"[env] torch_npu 不可用：{e}")
        print("[env] 本测试需要 NPU 环境（flexhead-tl conda env），退出。")
        return 1

    # 加载 sparse_attention_fwd
    try:
        sfa, source = _probe_sparse_attention_fwd()
    except ImportError as e:
        print(f"[env] sparse_attention_fwd import 失败：\n{e}")
        return 1
    _print_signature(sfa, source)

    if args.probe:
        print("\n[probe] done. 请把以上 signature 内容报回来。")
        return 0

    results = {}
    if args.case in ("bs", "all"):
        results["block-sparse"] = run_block_sparse_case(sfa, device)
    if args.case in ("sl", "all"):
        results["stream-llm"] = run_stream_llm_case(sfa, device)

    print("\n" + "=" * 60)
    print("[summary]")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\n  overall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
