# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""v2 启动复测：`npu_fusion_attention.sparse_mode` 行为表。

背景：CANN 8.1.RC1 下 sparse_mode=2/4 不按文档语义生效（详见
docs/context_checkpoint.md §9.1 / §9.3），导致 v1 路径 A 全程依赖
显式 [S_q, S_k] bool atten_mask，O(S²) 显存吃满 128K。

v2 升 CANN 8.2.RC1 后，本脚本逐项对照手写 PyTorch eager 参考，
确认 sparse_mode=2/4 是否修复，并直接决定 v2 路线分支：

  - 分支 A（修了）：路径 A / streaming 段 1 可省 mask，128K 解锁
  - 分支 B（没修）：mask 留着，v2 工程量翻倍（分块 / SP）

跑法：
    python tests/test_sparse_mode_quirk.py            # 标准
    python tests/test_sparse_mode_quirk.py -v         # 详细输出

退出码：3 个子测试全 PASS 退出 0，任一 FAIL 退出 1。
退出码不代表"成功 / 失败"，只代表"是否符合分支 A 的乐观预期"。
分支 B（FAIL）也是合法结果，按 docs/context_checkpoint.md §11.2 决策。
"""

import argparse
import math
import sys
import traceback

import torch


def _print_status(name: str, ok: bool, msg: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if msg:
        line += f"  ({msg})"
    print(line)


def _require_npu(verbose: bool) -> "torch.device":
    try:
        import torch_npu  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "torch_npu import failed — 当前环境不是昇腾 NPU 机器，或 CANN 未 source。"
            f" 原始错误：{e}"
        )
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() == False — 驱动 / 固件未就绪")
    dev = torch.device("npu:0")
    if verbose:
        import torch_npu

        print(
            f"    [env] torch={torch.__version__} torch_npu={torch_npu.__version__} "
            f"npu_count={torch.npu.device_count()} device={dev}"
        )
    return dev


def _eager_attention_ref(q, k, v, scale, mask_2d: torch.Tensor | None):
    """PyTorch eager 参考（fp32 中间，输出 cast 回输入 dtype）。

    mask_2d: [S_q, S_k] bool, True=屏蔽（NPU 惯例）；None 表示 full attention。
    """
    in_dtype = q.dtype
    qf = q.float()
    kf = k.float()
    vf = v.float()
    attn = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    if mask_2d is not None:
        attn = attn.masked_fill(mask_2d, float("-inf"))
    probs = torch.softmax(attn, dim=-1)
    out = torch.matmul(probs, vf)
    return out.to(in_dtype)


def _make_qkv(dev, dtype):
    B, N, S, D = 1, 4, 256, 128
    torch.manual_seed(0)
    q = torch.randn(B, N, S, D, dtype=dtype, device=dev)
    k = torch.randn(B, N, S, D, dtype=dtype, device=dev)
    v = torch.randn(B, N, S, D, dtype=dtype, device=dev)
    scale = 1.0 / math.sqrt(D)
    return q, k, v, scale, S


# ---------------------------------------------------------------------------
# 子测试 1：sparse_mode=2 无 mask vs 手写 causal
# ---------------------------------------------------------------------------


def test_sparse_mode_2_causal_no_mask(verbose: bool) -> bool:
    name = "sparse_mode=2 (no mask) should be causal"
    try:
        import torch_npu

        dev = _require_npu(verbose)
        q, k, v, scale, S = _make_qkv(dev, torch.float16)

        result = torch_npu.npu_fusion_attention(
            q, k, v,
            head_num=q.shape[1],
            input_layout="BNSD",
            scale=scale,
            sparse_mode=2,
        )
        out_npu = result[0] if isinstance(result, (tuple, list)) else result

        causal_mask = torch.ones(S, S, device=dev, dtype=torch.bool).triu(diagonal=1)
        out_ref = _eager_attention_ref(q, k, v, scale, causal_mask)

        diff = (out_npu.float() - out_ref.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        ok = max_abs < 1e-2

        if verbose:
            print(f"    [sm2] max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}  "
                  f"{'== causal (分支 A)' if ok else '!= causal (分支 B，仍是 full)'}")
        _print_status(name, ok, f"max_abs_diff={max_abs:.3e}")
        return ok

    except Exception as e:  # noqa: BLE001
        _print_status(name, False, f"exception: {e.__class__.__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# 子测试 2：sparse_mode=4 + pre_tockens/next_tockens 应为 sliding-window band
# ---------------------------------------------------------------------------


def test_sparse_mode_4_sliding_window(verbose: bool) -> bool:
    name = "sparse_mode=4 (pre_tockens=W-1, next_tockens=0) should be sliding-window"
    try:
        import torch_npu

        dev = _require_npu(verbose)
        q, k, v, scale, S = _make_qkv(dev, torch.float16)
        W = 64  # window size

        # 参考 band mask：每个 q[i] 只看 k[max(0,i-W+1)..i]，True=屏蔽
        idx_q = torch.arange(S, device=dev).unsqueeze(1)  # [S,1]
        idx_k = torch.arange(S, device=dev).unsqueeze(0)  # [1,S]
        band = (idx_k > idx_q) | (idx_k < idx_q - (W - 1))  # True=mask

        try:
            result = torch_npu.npu_fusion_attention(
                q, k, v,
                head_num=q.shape[1],
                input_layout="BNSD",
                scale=scale,
                sparse_mode=4,
                pre_tockens=W - 1,
                next_tockens=0,
            )
        except TypeError as e:
            # 某些版本拼写或参数名不同，记录后 FAIL
            raise RuntimeError(f"sparse_mode=4 调用签名不匹配：{e}")

        out_npu = result[0] if isinstance(result, (tuple, list)) else result
        out_ref = _eager_attention_ref(q, k, v, scale, band)

        diff = (out_npu.float() - out_ref.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        ok = max_abs < 1e-2

        # 同时对照 full attention，便于诊断"是不是又退化成 full"
        out_full = _eager_attention_ref(q, k, v, scale, None)
        max_vs_full = (out_npu.float() - out_full.float()).abs().max().item()

        if verbose:
            verdict = (
                "== sliding-window (分支 A)" if ok
                else f"!= band；vs full={max_vs_full:.3e}（≈0 表示退化成 full）"
            )
            print(f"    [sm4] W={W} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}  {verdict}")
        _print_status(name, ok, f"max_abs_diff={max_abs:.3e}")
        return ok

    except Exception as e:  # noqa: BLE001
        _print_status(name, False, f"exception: {e.__class__.__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# 子测试 3：sparse_mode=1 + 显式 mask（兜底，预期始终 PASS）
# ---------------------------------------------------------------------------


def test_sparse_mode_1_explicit_mask(verbose: bool) -> bool:
    name = "sparse_mode=1 + explicit causal mask (baseline)"
    try:
        import torch_npu

        dev = _require_npu(verbose)
        q, k, v, scale, S = _make_qkv(dev, torch.float16)
        causal_mask = torch.ones(S, S, device=dev, dtype=torch.bool).triu(diagonal=1)

        try:
            result = torch_npu.npu_fusion_attention(
                q, k, v,
                head_num=q.shape[1],
                input_layout="BNSD",
                scale=scale,
                sparse_mode=1,
                atten_mask=causal_mask,
            )
        except TypeError:
            result = torch_npu.npu_fusion_attention(
                q, k, v,
                head_num=q.shape[1],
                input_layout="BNSD",
                scale=scale,
                sparse_mode=1,
                atten_mask=causal_mask.to(torch.uint8),
            )
        out_npu = result[0] if isinstance(result, (tuple, list)) else result
        out_ref = _eager_attention_ref(q, k, v, scale, causal_mask)

        diff = (out_npu.float() - out_ref.float()).abs()
        max_abs = diff.max().item()
        ok = max_abs < 1e-2

        if verbose:
            print(f"    [sm1] max_abs={max_abs:.3e}  {'OK' if ok else 'baseline 居然挂了'}")
        _print_status(name, ok, f"max_abs_diff={max_abs:.3e}")
        return ok

    except Exception as e:  # noqa: BLE001
        _print_status(name, False, f"exception: {e.__class__.__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 启动：sparse_mode 行为复测")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("MInference-NPU v2 启动 — npu_fusion_attention.sparse_mode 行为复测")
    print("=" * 70)

    results = [
        test_sparse_mode_1_explicit_mask(args.verbose),
        test_sparse_mode_2_causal_no_mask(args.verbose),
        test_sparse_mode_4_sliding_window(args.verbose),
    ]

    print("-" * 70)
    n_pass = sum(1 for r in results if r)
    print(f"PASS: {n_pass}/{len(results)}")

    # baseline (sm1) 必须 PASS；sm2/sm4 PASS 数决定分支
    baseline_ok = results[0]
    sm2_ok = results[1]
    sm4_ok = results[2]

    if not baseline_ok:
        print("[ERROR] sm1 baseline 都挂了 — 环境 / API 异常，先排查 baseline 再说")
        return 1

    if sm2_ok and sm4_ok:
        print("[v2 路线 → 分支 A] sm2 和 sm4 都修复，路径 A / streaming 段 1 可去 mask，128K 直接解锁。")
        print("                   把结果回填 docs/context_checkpoint.md §9.1 / §9.3 + memory，开始批量回滚 mask。")
        return 0

    if sm2_ok or sm4_ok:
        print(f"[v2 路线 → 部分修复] sm2={'OK' if sm2_ok else 'NG'}, sm4={'OK' if sm4_ok else 'NG'}")
        print("                   能省一处算一处，剩下的走分支 B。")
        return 0  # 部分修复也算可推进

    print("[v2 路线 → 分支 B] sm2 / sm4 仍不按文档语义生效，mask 留着。")
    print("                   v2 仍需走 dense 分块 / 序列并行解决 O(S²) 显存。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
