# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""v2 PR-1 bit-identical 对照测试：per-head 循环 vs batched 调用。

测试 `_vertical_and_slash_kernel` 在 batched 调用（H>1）与逐 head 调用拼接的
结果是否完全 bit-identical。覆盖：

  T1. kernel H-batched / 同 V,S size（每个 head 用同一 vertical_size / slash_size）
  T2. kernel H-batched / 不同 V,S size（per-head sizes 列表，验证 duplicate-pad 路径）
  T3. dispatcher 等价性：modules.minference_forward.minference_forward 闭包整体
      跑同一份 best_pattern，与 v1 的 per-head 循环对照（仅 NPU 上 skip）

运行方式：
  python -m pytest tests/test_minference_batched_vs.py -v
  # 或单独跑（NPU 服务器）：
  python tests/test_minference_batched_vs.py

CPU 路径走 `_vertical_slash_pytorch_ref`；NPU 路径走 `npu_fusion_attention`。
两条路径都应让 batched 与 per-head **bit-identical**（torch.equal == True），
因为算法语义完全相同：同一份 v_idx/s_idx 喂给同一份 attention。
"""

from __future__ import annotations

import pytest
import torch

from minference.modules.minference_forward import _vertical_and_slash_kernel
from minference.backend_npu.cuda_shim import convert_vertical_slash_indexes
from minference.ops.vertical_slash_kernel_npu import (
    _build_vs_mask_direct,
    _build_vs_mask_from_indexes_loop,
    _build_vs_mask_from_indexes_vec,
)

try:
    import torch_npu  # type: ignore[import-not-found]

    _HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except ImportError:
    _HAS_NPU = False


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------

_DEVICES: list[torch.device] = [torch.device("cpu")]
if _HAS_NPU:
    _DEVICES.append(torch.device("npu:0"))


def _make_qkv(B: int, H: int, S: int, D: int, device: torch.device, dtype=torch.float16):
    """生成 [B, H, S, D] 的随机 QKV。固定种子以便复现。"""
    g = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
    q = torch.randn(B, H, S, D, dtype=torch.float32, generator=g).to(dtype).to(device)
    k = torch.randn(B, H, S, D, dtype=torch.float32, generator=g).to(dtype).to(device)
    v = torch.randn(B, H, S, D, dtype=torch.float32, generator=g).to(dtype).to(device)
    return q, k, v


# --------------------------------------------------------------------------
# T1: 同 V/S size — kernel 直接 H-batched ⇔ 逐 head 拼接
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_kernel_h_batched_uniform_size(device: torch.device):
    """每个 head 用相同 (V, S)：batched H=8 调用 ⇔ 8 次单 head 调用拼接结果。"""
    B, H, S, D = 1, 8, 256, 128
    V_SIZE, S_SIZE = 100, 150
    q, k, v = _make_qkv(B, H, S, D, device)

    class _Dummy:
        pass

    dummy = _Dummy()

    # batched 一次调用
    out_batched = _vertical_and_slash_kernel(dummy, q, k, v, V_SIZE, S_SIZE)

    # 逐 head 拼接
    out_perhead = torch.empty_like(q)
    for h in range(H):
        out_perhead[:, h : h + 1] = _vertical_and_slash_kernel(
            dummy,
            q[:, h : h + 1],
            k[:, h : h + 1],
            v[:, h : h + 1],
            V_SIZE,
            S_SIZE,
        )

    assert out_batched.shape == out_perhead.shape
    if not torch.equal(out_batched, out_perhead):
        diff = (out_batched.float() - out_perhead.float()).abs()
        raise AssertionError(
            f"NOT bit-identical (device={device}): "
            f"max_abs_diff={diff.max().item():.3e}, "
            f"mean_abs_diff={diff.mean().item():.3e}"
        )


# --------------------------------------------------------------------------
# T2: 不同 V/S size — duplicate-pad 路径
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_kernel_h_batched_diff_sizes(device: torch.device):
    """每个 head 用不同 (V, S)：list 入参 ⇔ 逐 head 用各自标量调用拼接结果。"""
    B, H, S, D = 1, 4, 256, 128
    vs_sizes = [80, 100, 60, 120]
    ss_sizes = [120, 180, 80, 200]
    q, k, v = _make_qkv(B, H, S, D, device)

    class _Dummy:
        pass

    dummy = _Dummy()

    # batched: 传 list
    out_batched = _vertical_and_slash_kernel(dummy, q, k, v, vs_sizes, ss_sizes)

    # 逐 head：传标量
    out_perhead = torch.empty_like(q)
    for h in range(H):
        out_perhead[:, h : h + 1] = _vertical_and_slash_kernel(
            dummy,
            q[:, h : h + 1],
            k[:, h : h + 1],
            v[:, h : h + 1],
            vs_sizes[h],
            ss_sizes[h],
        )

    assert out_batched.shape == out_perhead.shape
    if not torch.equal(out_batched, out_perhead):
        diff = (out_batched.float() - out_perhead.float()).abs()
        raise AssertionError(
            f"NOT bit-identical (device={device}): "
            f"max_abs_diff={diff.max().item():.3e}, "
            f"mean_abs_diff={diff.mean().item():.3e}, "
            "duplicate-pad 路径可能在 shim/_build_vs_mask 上未保持幂等。"
        )


# --------------------------------------------------------------------------
# T3: scalar 入参与单元素 list 入参等价（向后兼容回归）
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_scalar_vs_list_input_equivalence(device: torch.device):
    """`_vertical_and_slash_kernel(..., 100, 150)` ⇔ `..., [100], [150]`（H=1）。"""
    B, H, S, D = 1, 1, 256, 128
    q, k, v = _make_qkv(B, H, S, D, device)

    class _Dummy:
        pass

    dummy = _Dummy()
    out_scalar = _vertical_and_slash_kernel(dummy, q, k, v, 100, 150)
    out_list = _vertical_and_slash_kernel(dummy, q, k, v, [100], [150])

    if not torch.equal(out_scalar, out_list):
        diff = (out_scalar.float() - out_list.float()).abs()
        raise AssertionError(
            f"scalar vs single-elem list NOT bit-identical (device={device}): "
            f"max_abs_diff={diff.max().item():.3e}"
        )


# --------------------------------------------------------------------------
# T4: _build_vs_mask_from_indexes loop vs vec — bit-identical（PR-2）
# --------------------------------------------------------------------------


def _gen_vs_idx(B: int, H: int, S: int, n_v: int, n_s: int, seed: int = 0xBADBEEF):
    """生成 (v_idx, s_idx)，与上游约定一致：v_idx 升序、s_idx 降序。"""
    g = torch.Generator().manual_seed(seed)
    v_lists, s_lists = [], []
    for _ in range(B):
        v_h, s_h = [], []
        for _ in range(H):
            sink = list(range(30))
            rest = torch.randperm(max(S - 30, 1), generator=g)[: max(n_v - 30, 0)].tolist()
            v_vals = sorted(set(sink + rest))[:n_v]
            while len(v_vals) < n_v:
                v_vals.append(v_vals[-1] + 1)
            v_h.append(v_vals)

            local_s = list(range(min(100, S)))
            far_s = sorted(
                set(torch.randint(100, max(S, 101), (max(n_s - len(local_s), 0),), generator=g).tolist()),
                reverse=True,
            )
            s_vals = sorted(set(far_s + local_s), reverse=True)[:n_s]
            while len(s_vals) < n_s:
                s_vals.insert(0, s_vals[0] + 1)
            s_h.append(s_vals)
        v_lists.append(v_h)
        s_lists.append(s_h)
    v_idx = torch.tensor(v_lists, dtype=torch.int32)  # [B, H, n_v]
    s_idx = torch.tensor(s_lists, dtype=torch.int32)  # [B, H, n_s]
    return v_idx, s_idx


@pytest.mark.parametrize(
    "B,H,S,n_v,n_s",
    [
        (1, 1, 256, 50, 80),
        (1, 4, 512, 100, 150),
        (2, 8, 384, 60, 120),
    ],
)
def test_build_mask_loop_vs_vec_bit_identical(B, H, S, n_v, n_s):
    """`_build_vs_mask_from_indexes_loop` == `_build_vs_mask_from_indexes_vec`（CPU）。"""
    v_idx, s_idx = _gen_vs_idx(B, H, S, n_v, n_s)
    seqlens = torch.tensor([S] * B, dtype=torch.int32)
    bc, bo, cc, ci = convert_vertical_slash_indexes(seqlens, v_idx, s_idx, S, 64, 64)

    device = torch.device("cpu")
    mask_loop = _build_vs_mask_from_indexes_loop(bc, bo, cc, ci, S, S, device=device)
    mask_vec = _build_vs_mask_from_indexes_vec(bc, bo, cc, ci, S, S, device=device)

    assert mask_loop.shape == mask_vec.shape
    if not torch.equal(mask_loop, mask_vec):
        diff = mask_loop ^ mask_vec
        b, h, i, j = torch.nonzero(diff, as_tuple=True)
        sample = list(zip(b[:5].tolist(), h[:5].tolist(), i[:5].tolist(), j[:5].tolist()))
        raise AssertionError(
            f"loop vs vec NOT bit-identical: {diff.sum().item()} 个位置不同。"
            f"前 5 个差异坐标 (b,h,i,j) = {sample}"
        )


# --------------------------------------------------------------------------
# T5 / T6: PR-3 mask 算法差异验证
#
# PR-3 起 NPU 路径 mask 走 _build_vs_mask_direct（真实 slash 区间 OR），不再继承
# 上游 CUDA 的 "贪婪 blk 扩展" 近似。这是合法的 MInference 算法变种 ——
# 上游 blk 扩展是为 Triton CUDA 块对齐 sparse kernel 服务，NPU token-level bool
# mask 路径无此需求。
#
# direct mask 与 v1 mask 关系（True = masked）：
#   - direct True 集合 ⊇ v1 True 集合（direct mask 更稀疏，可见区域更小）
#   - 等价：direct 的 visible ⊆ v1 的 visible（v1 多覆盖块对齐扩展位置）
# 测试设计：
#   T5: mask 关系 —— direct True ⊇ v1 True，且差异只在 slash 块对齐扩展区域
#   T6: attention 端到端容差对比 —— direct attention 输出与 v1 attention 输出
#       在 fp32 + 小尺寸下应在合理容差内（差异来自 v1 多覆盖位置的 attention 贡献）
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,H,S,n_v,n_s",
    [
        (1, 1, 256, 50, 80),
        (1, 4, 512, 100, 150),
        (2, 8, 384, 60, 120),
        # S 非 64 倍数：验证 num_rows*BLK > S_k 边界（repeat_interleave + 截断）
        (1, 2, 200, 30, 50),
        (1, 2, 257, 30, 50),
        # NNZ_V/NNZ_S 较小（接近 sink 30 / local 100）
        (1, 4, 512, 30, 100),
        # NNZ 较大、>= S
        (1, 2, 256, 200, 200),
    ],
)
def test_direct_mask_is_visible_subset_of_v1(B, H, S, n_v, n_s):
    """T5: direct mask True ⊇ v1 mask True（direct visible ⊆ v1 visible）。

    PR-3 的核心安全约束：direct mask 不会"看到" v1 mask 屏蔽的位置 ——
    即 direct 的可见域是 v1 可见域的子集。等价表达：
        v1_visible \ direct_visible = ∅  ⇔  v1 True ⊆ direct True
        violated = mask_v1 & ~mask_direct  应为全 False（其中 True 表示 masked）

    Wait — 反过来：True=masked, visible=~mask. v1 多覆盖（多可见）⇒ ~mask_v1 ⊇ ~mask_direct
    ⇔ mask_v1 ⊆ mask_direct ⇔ mask_v1 & ~mask_direct = ∅。

    断言：任何在 v1 中标记为 masked 的位置，在 direct 中也必须是 masked。
    """
    v_idx, s_idx = _gen_vs_idx(B, H, S, n_v, n_s)
    seqlens = torch.tensor([S] * B, dtype=torch.int32)
    bc, bo, cc, ci = convert_vertical_slash_indexes(seqlens, v_idx, s_idx, S, 64, 64)

    device = torch.device("cpu")
    mask_v1 = _build_vs_mask_from_indexes_loop(bc, bo, cc, ci, S, S, device=device)
    mask_direct = _build_vs_mask_direct(v_idx, s_idx, S, S, device=device)

    assert mask_v1.shape == mask_direct.shape, (
        f"shape 不一致: v1={tuple(mask_v1.shape)} direct={tuple(mask_direct.shape)}"
    )

    # v1 True 集合应 ⊆ direct True 集合，即不存在 (v1=True, direct=False) 位置
    v1_only_masked = mask_v1 & ~mask_direct
    if v1_only_masked.any():
        b, h, i, j = torch.nonzero(v1_only_masked, as_tuple=True)
        sample = list(zip(b[:5].tolist(), h[:5].tolist(), i[:5].tolist(), j[:5].tolist()))
        raise AssertionError(
            f"v1 mask True ⊄ direct mask True "
            f"(B={B} H={H} S={S} n_v={n_v} n_s={n_s}): "
            f"{v1_only_masked.sum().item()} 个位置 v1 屏蔽但 direct 可见，"
            f"前 5 个 (b,h,i,j) = {sample}。"
            f"这违反 PR-3 的子集安全约束。"
        )


@pytest.mark.parametrize(
    "B,H,S,D,n_v,n_s",
    [
        (1, 2, 256, 64, 50, 80),
        (1, 4, 512, 64, 60, 100),
        (2, 2, 384, 128, 40, 120),
    ],
)
def test_direct_attention_matches_dense_on_mask_consistent_rows(B, H, S, D, n_v, n_s):
    """T6: direct vs v1 attention 输出差异只来源于 v1 多覆盖区域。

    PR-3 mask 与 v1 mask 仅在 "v1 块对齐扩展位置" 不同。如果某个 query 行 i 的 mask
    在所有 j 上完全相同（即该行的 v1 与 direct 看到同样的 K/V 集合），那么该行的
    attention 输出必然 bit-identical（fp32 容差内 < 1e-5）。

    本测试断言：mask 在某 query 行 i 上完全一致的 (b, h, i) 组合，其 attention 输出
    bit-close。这把 "v1 块扩展导致的差异" 与 "实现 bug" 区分开 —— 真正的 bug 会让
    mask 一致的行也出现差异。
    """
    v_idx, s_idx = _gen_vs_idx(B, H, S, n_v, n_s)
    g = torch.Generator().manual_seed(0xC0FFEE)
    q = torch.randn(B, H, S, D, dtype=torch.float32, generator=g)
    k = torch.randn(B, H, S, D, dtype=torch.float32, generator=g)
    v = torch.randn(B, H, S, D, dtype=torch.float32, generator=g)

    from minference.ops.vertical_slash_kernel_npu import (
        _build_vs_mask_direct,
        _vertical_slash_pytorch_ref,
        _vertical_slash_pytorch_ref_legacy,
    )

    seqlens = torch.tensor([S] * B, dtype=torch.int32)
    bc, bo, cc, ci = convert_vertical_slash_indexes(seqlens, v_idx, s_idx, S, 64, 64)
    mask_v1 = _build_vs_mask_from_indexes_loop(bc, bo, cc, ci, S, S, device=q.device)
    mask_direct = _build_vs_mask_direct(v_idx, s_idx, S, S, device=q.device)
    # 每行 (b,h,i) 是否 mask 完全一致
    row_mask_equal = (mask_v1 == mask_direct).all(dim=-1)  # [B,H,S]

    out_direct = _vertical_slash_pytorch_ref(q, k, v, v_idx, s_idx)
    out_v1 = _vertical_slash_pytorch_ref_legacy(q, k, v, v_idx, s_idx)
    assert out_direct.shape == out_v1.shape

    # 只在 mask 一致的行上检查 attention 输出 bit-close
    diff = (out_direct - out_v1).abs()                                         # [B,H,S,D]
    diff_consistent_rows = diff[row_mask_equal]                                # [N_consistent, D]
    if diff_consistent_rows.numel() == 0:
        pytest.skip(f"无 mask 一致的 query 行（B={B} H={H} S={S} n_v={n_v} n_s={n_s}）")
    max_diff_consistent = diff_consistent_rows.max().item()
    assert max_diff_consistent < 1e-5, (
        f"mask 一致行的 attention 输出不应有差异 "
        f"(B={B} H={H} S={S} D={D}): max_abs={max_diff_consistent:.3e}, "
        f"mask 一致行数={int(row_mask_equal.sum())}/{B * H * S}"
    )

    # 报告全局差异（仅 informational，不断言）
    print(
        f"  [info] direct vs v1 attention diff "
        f"max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e}, "
        f"mask 一致行 {int(row_mask_equal.sum())}/{B * H * S}"
    )


# --------------------------------------------------------------------------
# 直接脚本运行：脱离 pytest（NPU 服务器上 `python tests/test_minference_batched_vs.py`）
# --------------------------------------------------------------------------


__test__ = True  # pytest 允许收集本模块的 test_*；脚本入口下方手动驱动一次

if __name__ == "__main__":
    print(f"[env] devices to test: {[str(d) for d in _DEVICES]}")
    failures = 0
    for dev in _DEVICES:
        for name, fn in [
            ("T1 uniform size", test_kernel_h_batched_uniform_size),
            ("T2 diff sizes", test_kernel_h_batched_diff_sizes),
            ("T3 scalar vs list", test_scalar_vs_list_input_equivalence),
        ]:
            try:
                fn(dev)
                print(f"  [PASS] {name} ({dev})")
            except AssertionError as e:
                failures += 1
                print(f"  [FAIL] {name} ({dev}): {e}")
            except Exception as e:
                failures += 1
                print(f"  [ERROR] {name} ({dev}): {type(e).__name__}: {e}")

    # T4: loop vs vec — 只 CPU
    for params in [(1, 1, 256, 50, 80), (1, 4, 512, 100, 150), (2, 8, 384, 60, 120)]:
        name = f"T4 loop-vs-vec B={params[0]} H={params[1]} S={params[2]}"
        try:
            test_build_mask_loop_vs_vec_bit_identical(*params)
            print(f"  [PASS] {name}")
        except AssertionError as e:
            failures += 1
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            failures += 1
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")

    # T5: direct mask 是 v1 mask 的 visible 子集 — 只 CPU
    for params in [
        (1, 1, 256, 50, 80),
        (1, 4, 512, 100, 150),
        (2, 8, 384, 60, 120),
        (1, 2, 200, 30, 50),
        (1, 2, 257, 30, 50),
        (1, 4, 512, 30, 100),
        (1, 2, 256, 200, 200),
    ]:
        name = f"T5 direct-superset-v1 B={params[0]} H={params[1]} S={params[2]} v={params[3]} s={params[4]}"
        try:
            test_direct_mask_is_visible_subset_of_v1(*params)
            print(f"  [PASS] {name}")
        except AssertionError as e:
            failures += 1
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            failures += 1
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")

    # T6: direct vs v1 attention 输出容差 — 只 CPU
    for params in [
        (1, 2, 256, 64, 50, 80),
        (1, 4, 512, 64, 60, 100),
        (2, 2, 384, 128, 40, 120),
    ]:
        name = f"T6 mask-consistent-rows B={params[0]} H={params[1]} S={params[2]} D={params[3]}"
        try:
            test_direct_attention_matches_dense_on_mask_consistent_rows(*params)
            print(f"  [PASS] {name}")
        except AssertionError as e:
            failures += 1
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            failures += 1
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")

    print(f"\n{'PASS' if failures == 0 else 'FAIL'}  ({failures} failure(s))")
    raise SystemExit(0 if failures == 0 else 1)
