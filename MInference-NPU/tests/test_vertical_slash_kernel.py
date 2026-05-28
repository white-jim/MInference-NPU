# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""M4 vertical-slash kernel 单元测试。

包含以下测试组：
  T1. convert_vertical_slash_indexes — 基本格式与计数正确性
  T2. convert_vertical_slash_indexes — 因果一致性（block 内不含 future KV）
  T3. _build_vs_mask_from_indexes — mask 形状与边界
  T4. vertical_slash_sparse_attention (ref) — 与 causal dense 数值接近（高稀疏度下退化）
  T5. vertical_slash_sparse_attention (ref) — 满覆盖时与 causal dense 完全一致
  T6. head_dim pad — 非标准 head_dim 自动 pad，输出 shape 正确
  T7. NPU vs ref — 仅在 NPU 环境跑（否则 skip）
  T8. 参数扫描 — 不同 (S, NNZ_V, NNZ_S, block_size) 组合不 crash

运行方式：
  python -m pytest tests/test_vertical_slash_kernel.py -v
  # NPU 环境额外跑 T7：
  python -m pytest tests/test_vertical_slash_kernel.py -v -k "npu"
"""

from __future__ import annotations

import math
import pytest
import torch

# --------------------------------------------------------------------------
# 导入被测模块
# --------------------------------------------------------------------------
from minference.backend_npu.cuda_shim import convert_vertical_slash_indexes
from minference.ops.vertical_slash_kernel_npu import (
    _build_vs_mask_from_indexes,
    _vertical_slash_pytorch_ref,
    vertical_slash_sparse_attention,
)

try:
    import torch_npu  # type: ignore[import-not-found]
    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------

def _make_v_s_idx(q_len: int, n_v: int, n_s: int, device="cpu"):
    """生成合法的 v_idx（升序）和 s_idx（降序）。

    强制保留前 30 个 sink 列（v_idx 前 30 为 0..29）和
    最近 100 条 local slash（s_idx 末 100 为 0..99）。
    """
    # v_idx: 前 30 sink + 随机 topk，升序，无重复
    sink = list(range(30))
    rest = sorted(torch.randperm(max(q_len - 30, 1))[:max(n_v - 30, 0)].tolist())
    v_vals = sorted(set(sink + rest))[:n_v]
    while len(v_vals) < n_v:
        v_vals.append(v_vals[-1] + 1 if v_vals else 0)
    v_idx = torch.tensor([v_vals], dtype=torch.int32)  # [1, n_v]

    # s_idx: 最近 100 local (0..99) + 随机远端，降序
    local_s = list(range(min(100, q_len)))
    far_s = sorted(
        torch.randint(100, max(q_len, 101), (max(n_s - len(local_s), 0),)).unique().tolist(),
        reverse=True,
    )
    s_vals = sorted(set(far_s + local_s), reverse=True)[:n_s]
    while len(s_vals) < n_s:
        s_vals.insert(0, s_vals[0] + 1 if s_vals else 0)
    s_idx = torch.tensor([s_vals], dtype=torch.int32)  # [1, n_s]

    # 增加 batch 和 head 维度 → [B=1, H=1, ...]
    return v_idx.unsqueeze(0).to(device), s_idx.unsqueeze(0).to(device)


def _causal_dense_ref(q, k, v):
    """因果 dense attention（PyTorch）作为黄金参考。"""
    B, H, S_q, D = q.shape
    S_k = k.shape[2]
    scale = D ** -0.5
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # [B,H,S_q,S_k]
    i_idx = torch.arange(S_q, device=q.device)
    j_idx = torch.arange(S_k, device=q.device)
    causal_mask = j_idx[None, None, None, :] > i_idx[None, None, :, None]
    logits.masked_fill_(causal_mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, v.float()).to(q.dtype)


# --------------------------------------------------------------------------
# T1: convert_vertical_slash_indexes 基本格式
# --------------------------------------------------------------------------

class TestConvertVSIndexes:
    def test_output_shapes(self):
        q_len = 256
        n_v, n_s = 64, 128
        v_idx, s_idx = _make_v_s_idx(q_len, n_v, n_s)
        seqlens = torch.tensor([q_len], dtype=torch.int32)

        bc, bo, cc, ci = convert_vertical_slash_indexes(
            seqlens, v_idx, s_idx, q_len, 64, 64
        )

        num_rows = q_len // 64  # = 4
        assert bc.shape == (1, 1, num_rows), f"block_count shape: {bc.shape}"
        assert bo.shape == (1, 1, num_rows, n_s), f"block_offset shape: {bo.shape}"
        assert cc.shape == (1, 1, num_rows), f"column_count shape: {cc.shape}"
        assert ci.shape == (1, 1, num_rows, n_v), f"column_index shape: {ci.shape}"

    def test_counts_non_negative(self):
        q_len = 512
        v_idx, s_idx = _make_v_s_idx(q_len, 100, 200)
        seqlens = torch.tensor([q_len], dtype=torch.int32)
        bc, bo, cc, ci = convert_vertical_slash_indexes(seqlens, v_idx, s_idx, q_len, 64, 64)

        assert (bc >= 0).all(), "block_count 含负值"
        assert (bc <= s_idx.shape[-1]).all(), "block_count 超出 NNZ_S"
        assert (cc >= 0).all(), "column_count 含负值"
        assert (cc <= v_idx.shape[-1]).all(), "column_count 超出 NNZ_V"

    def test_block_count_increases_with_row(self):
        """更晚的 query block 通常有更多 KV 块（因果方向）。"""
        q_len = 512
        v_idx, s_idx = _make_v_s_idx(q_len, 50, 100)
        seqlens = torch.tensor([q_len], dtype=torch.int32)
        bc, _, _, _ = convert_vertical_slash_indexes(seqlens, v_idx, s_idx, q_len, 64, 64)
        # block_count[0] ≤ block_count[-1]（因果：早期 query block 能看的 KV block 更少）
        assert bc[0, 0, 0] <= bc[0, 0, -1], (
            f"首个 query block 的 block_count ({bc[0,0,0]}) 超过末尾 ({bc[0,0,-1]})"
        )

    def test_seqlen_shorter_than_context(self):
        """seqlen < context_size 时，多余的 query block 应全为 0。"""
        context_size = 256
        seqlen = 128
        v_idx, s_idx = _make_v_s_idx(seqlen, 30, 50)
        seqlens = torch.tensor([seqlen], dtype=torch.int32)
        bc, _, cc, _ = convert_vertical_slash_indexes(
            seqlens, v_idx, s_idx, context_size, 64, 64
        )
        # rows 2, 3 (start_m = 128, 192) >= seqlen=128，应全为 0
        assert bc[0, 0, 2] == 0 and bc[0, 0, 3] == 0
        assert cc[0, 0, 2] == 0 and cc[0, 0, 3] == 0


# --------------------------------------------------------------------------
# T2: convert_vertical_slash_indexes — 因果一致性
# --------------------------------------------------------------------------

class TestConvertVSCausal:
    def test_block_offsets_causal(self):
        """block_offset 的起始位置不超过 end_m（因果）。"""
        q_len = 384
        v_idx, s_idx = _make_v_s_idx(q_len, 60, 120)
        seqlens = torch.tensor([q_len], dtype=torch.int32)
        bc, bo, _, _ = convert_vertical_slash_indexes(seqlens, v_idx, s_idx, q_len, 64, 64)

        for bq in range(q_len // 64):
            end_m = (bq + 1) * 64
            n = int(bc[0, 0, bq].item())
            for bi in range(n):
                start = int(bo[0, 0, bq, bi].item())
                assert start < end_m, (
                    f"block_offset[bq={bq}, bi={bi}]={start} >= end_m={end_m}（违反因果）"
                )

    def test_column_index_values_in_range(self):
        """column_index 值应在 [0, q_len) 范围内。"""
        q_len = 256
        v_idx, s_idx = _make_v_s_idx(q_len, 50, 80)
        seqlens = torch.tensor([q_len], dtype=torch.int32)
        _, _, cc, ci = convert_vertical_slash_indexes(seqlens, v_idx, s_idx, q_len, 64, 64)

        for bq in range(q_len // 64):
            n = int(cc[0, 0, bq].item())
            for ci_i in range(n):
                col = int(ci[0, 0, bq, ci_i].item())
                assert 0 <= col < q_len, f"column_index[bq={bq}]={col} 超出范围 [0,{q_len})"


# --------------------------------------------------------------------------
# T3: mask 形状与边界
# --------------------------------------------------------------------------

class TestBuildMask:
    def _make_mask(self, S, n_v=50, n_s=100):
        v_idx, s_idx = _make_v_s_idx(S, n_v, n_s)
        seqlens = torch.tensor([S], dtype=torch.int32)
        bc, bo, cc, ci = convert_vertical_slash_indexes(
            seqlens, v_idx, s_idx, S, 64, 64
        )
        mask = _build_vs_mask_from_indexes(bc, bo, cc, ci, S, S, device=torch.device("cpu"))
        return mask, v_idx, s_idx

    def test_mask_shape(self):
        S = 256
        mask, _, _ = self._make_mask(S)
        assert mask.shape == (1, 1, S, S), f"mask shape: {mask.shape}"

    def test_mask_dtype_bool(self):
        mask, _, _ = self._make_mask(128)
        assert mask.dtype == torch.bool

    def test_upper_triangle_fully_masked(self):
        """严格上三角（j > i）必须全部为 True（masked），即因果约束完整。"""
        S = 256
        mask, _, _ = self._make_mask(S)
        m = mask[0, 0]  # [S, S]
        # 上三角（不含主对角）
        i_idx = torch.arange(S)
        j_idx = torch.arange(S)
        upper = j_idx[None, :] > i_idx[:, None]
        assert m[upper].all(), "上三角含未被 masked 的位置（未来 token 可见，违反因果）"

    def test_diagonal_not_fully_masked(self):
        """主对角线（j == i）至少有一些位置未 masked（当前 token 可见）。"""
        S = 256
        mask, _, _ = self._make_mask(S, n_v=50, n_s=200)
        m = mask[0, 0]
        diag = m.diagonal()
        assert (~diag).any(), "主对角线全被 masked（当前 token 不可见，不正确）"

    def test_sink_columns_visible(self):
        """前 30 个 sink 列对所有 query row（row >= col）应可见（因 v_idx 强制保留）。"""
        S = 256
        mask, _, _ = self._make_mask(S, n_v=50, n_s=100)
        m = mask[0, 0]  # [S, S]
        for col in range(30):
            # row >= col 的位置应为 False（未 masked）
            rows_ok = m[col:, col]
            assert (~rows_ok).all(), (
                f"sink 列 {col} 在 row>={col} 有被 masked 的位置（共 {rows_ok.sum().item()} 个）"
            )


# --------------------------------------------------------------------------
# T4: vertical_slash_sparse_attention (ref) — 与 dense 数值接近
# --------------------------------------------------------------------------

class TestVSRefVsDense:
    @pytest.mark.parametrize("S,dtype", [
        (256, torch.float32),
        (384, torch.float16),
        (512, torch.float32),
    ])
    def test_dense_upper_bound(self, S, dtype):
        """当 v_idx 和 s_idx 覆盖全部 KV 时，ref 输出应与 causal dense 完全一致。"""
        torch.manual_seed(42)
        B, H, D = 1, 1, 64
        q = torch.randn(B, H, S, D, dtype=dtype)
        k = torch.randn(B, H, S, D, dtype=dtype)
        v = torch.randn(B, H, S, D, dtype=dtype)

        # 全覆盖：vertical = 所有列（升序），slash = 所有行（降序）
        v_idx = torch.arange(S, dtype=torch.int32).unsqueeze(0).unsqueeze(0)  # [1,1,S]
        s_idx = torch.arange(S - 1, -1, -1, dtype=torch.int32).unsqueeze(0).unsqueeze(0)  # [1,1,S]

        out_ref = _vertical_slash_pytorch_ref(q, k, v, v_idx, s_idx)
        out_dense = _causal_dense_ref(q, k, v)

        atol = 1e-3 if dtype == torch.float32 else 5e-2
        diff = (out_ref - out_dense).abs().max().item()
        assert diff < atol, f"全覆盖时 ref 与 dense 差异 {diff:.4e} > atol={atol} (S={S}, dtype={dtype})"

    @pytest.mark.parametrize("S", [256, 512])
    def test_sparse_shape_correct(self, S):
        """稀疏模式下输出形状正确。"""
        torch.manual_seed(0)
        B, H, D = 1, 1, 64
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        v_idx, s_idx = _make_v_s_idx(S, 100, 200)

        out = _vertical_slash_pytorch_ref(q, k, v, v_idx, s_idx)
        assert out.shape == (B, H, S, D), f"输出形状错误：{out.shape}"
        assert not out.isnan().any(), "输出含 NaN"
        assert not out.isinf().any(), "输出含 Inf"


# --------------------------------------------------------------------------
# T5: 满覆盖 → 与 causal dense 完全等价
# --------------------------------------------------------------------------

class TestFullCoverage:
    def test_full_coverage_equals_dense(self):
        torch.manual_seed(7)
        B, H, S, D = 1, 1, 128, 64
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        v_idx = torch.arange(S, dtype=torch.int32)[None, None, :]
        s_idx = torch.arange(S - 1, -1, -1, dtype=torch.int32)[None, None, :]

        out = vertical_slash_sparse_attention(q, k, v, v_idx, s_idx)
        ref = _causal_dense_ref(q, k, v)

        diff = (out.float() - ref.float()).abs().max().item()
        assert diff < 1e-3, f"满覆盖差异 {diff:.4e}"


# --------------------------------------------------------------------------
# T6: head_dim pad
# --------------------------------------------------------------------------

class TestHeadDimPad:
    @pytest.mark.parametrize("D", [48, 96, 160])
    def test_nonstandard_headdim(self, D):
        torch.manual_seed(1)
        B, H, S = 1, 1, 128
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        v_idx, s_idx = _make_v_s_idx(S, 50, 80)

        out = vertical_slash_sparse_attention(q, k, v, v_idx, s_idx)
        assert out.shape == (B, H, S, D), f"head_dim pad 后输出 shape {out.shape} != {(B,H,S,D)}"
        assert not out.isnan().any(), "head_dim pad 后含 NaN"


# --------------------------------------------------------------------------
# T7: NPU vs ref
# --------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_NPU, reason="需要 torch_npu 和 NPU 设备")
class TestNPUvsRef:
    def test_npu_vs_ref_small(self):
        torch.manual_seed(42)
        B, H, S, D = 1, 1, 256, 64
        q = torch.randn(B, H, S, D, dtype=torch.float16).npu()
        k = torch.randn(B, H, S, D, dtype=torch.float16).npu()
        v = torch.randn(B, H, S, D, dtype=torch.float16).npu()
        v_idx, s_idx = _make_v_s_idx(S, 100, 200, device="npu")

        out_npu = vertical_slash_sparse_attention(q, k, v, v_idx, s_idx)
        out_ref = _vertical_slash_pytorch_ref(
            q.cpu().float(), k.cpu().float(), v.cpu().float(),
            v_idx.cpu(), s_idx.cpu()
        ).half()

        diff = (out_npu.cpu().float() - out_ref.float()).abs().max().item()
        assert diff < 1e-2, f"NPU vs ref 差异 {diff:.4e}"


# --------------------------------------------------------------------------
# T8: 参数扫描
# --------------------------------------------------------------------------

class TestParamSweep:
    @pytest.mark.parametrize("S,n_v,n_s", [
        (128, 30, 50),
        (256, 100, 200),
        (384, 200, 400),
        (512, 300, 500),
    ])
    def test_no_crash(self, S, n_v, n_s):
        torch.manual_seed(S)
        B, H, D = 1, 1, 64
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        v_idx, s_idx = _make_v_s_idx(S, n_v, n_s)

        out = vertical_slash_sparse_attention(q, k, v, v_idx, s_idx)
        assert out.shape == (B, H, S, D)
        assert not out.isnan().any()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_dtype_passthrough(self, dtype):
        torch.manual_seed(2)
        B, H, S, D = 1, 1, 128, 64
        q = torch.randn(B, H, S, D, dtype=dtype)
        k = torch.randn(B, H, S, D, dtype=dtype)
        v = torch.randn(B, H, S, D, dtype=dtype)
        v_idx, s_idx = _make_v_s_idx(S, 50, 80)

        out = vertical_slash_sparse_attention(q, k, v, v_idx, s_idx)
        assert out.dtype == dtype, f"输出 dtype {out.dtype} != 输入 dtype {dtype}"
