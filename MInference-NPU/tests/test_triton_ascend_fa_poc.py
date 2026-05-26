# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""PR-4-poc 验收脚本：triton-ascend dense FA kernel vs npu_fusion_attention。

这是 PR-4-poc 的 gate — 决定能否启动 PR-4-VS（真稀疏 VS kernel）。
跑两组评估：

  1. 精度对照（小尺寸 256/1024/2048，单 head / 4 head，fp16 causal）
     验收：max_abs_diff < 1e-2（fp16 softmax 噪声范围）

  2. 性能对照（8K / 16K，模拟 Llama-3 8B prefill：4 head × head_dim 128）
     验收：triton-ascend < 3× npu_fusion_attention（high-level DSL 上限合理）
     不要求打平 AscendC — 那是 PR-4-VS 之后才考虑的事

跑法：
    python tests/test_triton_ascend_fa_poc.py            # 精度 + 性能全跑
    python tests/test_triton_ascend_fa_poc.py -v         # 详细输出
    python tests/test_triton_ascend_fa_poc.py --skip-bench  # 仅精度（CI 用）

退出码：所有精度子测试 PASS 退出 0（性能只 print 不影响退出码 — 性能数据
给人看，决策权在人）。

非 pytest 脚本：
    __test__ = False，避免 pytest 收集（与 test_env.py / test_sparse_mode_quirk.py
    一致约定）。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback

import torch

__test__ = False
collect_ignore = ["test_triton_ascend_fa_poc.py"]


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _print_status(name: str, ok: bool, msg: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if msg:
        line += f"  ({msg})"
    print(line)


def _require_npu(verbose: bool):
    try:
        import torch_npu  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"torch_npu 未安装 / 当前不是 NPU 机器。原始：{e}"
        )
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() == False")
    dev = torch.device("npu:0")
    if verbose:
        print(
            f"    [env] torch={torch.__version__} "
            f"npu_count={torch.npu.device_count()} device={dev}"
        )
    return dev


def _npu_fa_dense_causal(q, k, v, scale):
    """Baseline：npu_fusion_attention dense causal（sparse_mode=1 + 显式 bool mask）。

    与 [[project-minference-npu-fa-sparse-mode-quirk]] 一致：sparse_mode=2/4 在
    CANN 8.1/8.2/8.5 三连未修复，dense causal 永远走 sparse_mode=1 + 显式 mask。
    """
    import torch_npu

    B, N, S_q, D = q.shape
    S_k = k.shape[2]
    causal_mask = torch.ones(S_q, S_k, device=q.device, dtype=torch.bool).triu(diagonal=1)
    try:
        result = torch_npu.npu_fusion_attention(
            q, k, v,
            head_num=N,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,
            atten_mask=causal_mask,
        )
    except TypeError:
        result = torch_npu.npu_fusion_attention(
            q, k, v,
            head_num=N,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=1,
            atten_mask=causal_mask.to(torch.uint8),
        )
    return result[0] if isinstance(result, (tuple, list)) else result


def _eager_ref_fp32(q, k, v, scale):
    """fp32 PyTorch eager 参考（精度黄金标准）。仅用于小尺寸（S<=2048）。"""
    qf = q.float()
    kf = k.float()
    vf = v.float()
    S_q = qf.shape[-2]
    S_k = kf.shape[-2]
    attn = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    mask = torch.ones(S_q, S_k, device=q.device, dtype=torch.bool).tril()
    attn = attn.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(attn, dim=-1)
    return torch.matmul(probs, vf).to(q.dtype)


# ---------------------------------------------------------------------------
# 精度测试
# ---------------------------------------------------------------------------


def test_precision(verbose: bool = False) -> bool:
    """小尺寸精度对照：triton-ascend kernel vs npu_fusion_attention vs fp32 eager。"""
    name = "test_precision"
    try:
        from minference.ops.triton_ascend_fa_poc import (
            has_triton,
            triton_ascend_fa_dense,
        )

        if not has_triton():
            _print_status(name, False, "triton-ascend 不可用")
            return False

        dev = _require_npu(verbose)

        # (B, N, S, D)
        cases = [
            (1, 1, 256, 128),
            (1, 4, 256, 128),
            (1, 4, 1024, 128),
            (2, 4, 1024, 128),
            (1, 4, 2048, 128),
        ]
        all_ok = True
        torch.manual_seed(0)
        for B, N, S, D in cases:
            q = torch.randn(B, N, S, D, dtype=torch.float16, device=dev)
            k = torch.randn(B, N, S, D, dtype=torch.float16, device=dev)
            v = torch.randn(B, N, S, D, dtype=torch.float16, device=dev)
            scale = 1.0 / math.sqrt(D)

            out_triton = triton_ascend_fa_dense(q, k, v, sm_scale=scale, causal=True)
            out_npu = _npu_fa_dense_causal(q, k, v, scale)
            out_ref = _eager_ref_fp32(q, k, v, scale)

            d_tn = (out_triton.float() - out_npu.float()).abs().max().item()
            d_tr = (out_triton.float() - out_ref.float()).abs().max().item()
            d_nr = (out_npu.float() - out_ref.float()).abs().max().item()

            # 验收：triton 与 fp32 ref 差距 < 1e-2（fp16 softmax 累积上限）
            ok = d_tr < 1e-2
            all_ok = all_ok and ok
            if verbose or not ok:
                print(
                    f"    [shape B={B} N={N} S={S} D={D}]"
                    f" tri-vs-npu={d_tn:.3e}"
                    f" tri-vs-ref={d_tr:.3e}"
                    f" npu-vs-ref={d_nr:.3e}"
                    f" {'OK' if ok else 'FAIL'}"
                )

        _print_status(name, all_ok)
        return all_ok

    except Exception as e:  # noqa: BLE001
        _print_status(name, False, f"exception: {e.__class__.__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# 性能 benchmark（print only，不影响退出码）
# ---------------------------------------------------------------------------


def _sync():
    torch.npu.synchronize()


def _bench_one(fn, q, k, v, scale, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        _ = fn(q, k, v, scale)
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(q, k, v, scale)
    _sync()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000.0  # ms / iter


def run_block_sweep(verbose: bool = False) -> None:
    """8K dense causal 上的 (BLOCK_M, BLOCK_N) sweep — 找最优 tile。

    诊断 PR-4-poc 首跑慢得离谱时（ratio > 50）用：BLOCK 大小是 triton-ascend 在 NPU
    上最主要的旋钮，64×64 是 GPU 默认值但对 NPU AI Core 经常太小。
    """
    try:
        from minference.ops.triton_ascend_fa_poc import (
            has_triton,
            triton_ascend_fa_dense,
        )

        if not has_triton():
            print("[sweep] SKIP — triton-ascend 不可用")
            return

        dev = _require_npu(verbose)

        B, N, S, D = 1, 4, 8192, 128
        torch.manual_seed(0)
        q = torch.randn(B, N, S, D, dtype=torch.float16, device=dev)
        k = torch.randn(B, N, S, D, dtype=torch.float16, device=dev)
        v = torch.randn(B, N, S, D, dtype=torch.float16, device=dev)
        scale = 1.0 / math.sqrt(D)

        # 先取 npu_fa baseline
        try:
            t_npu = _bench_one(_npu_fa_dense_causal, q, k, v, scale)
        except Exception as e:  # noqa: BLE001
            print(f"[sweep] npu_fa baseline FAIL: {e}")
            return

        block_pairs = [
            (32, 32),
            (64, 64),    # 默认
            (64, 128),
            (128, 64),
            (128, 128),
            (128, 256),
            (256, 128),
            (256, 256),
        ]

        print("-" * 72)
        print(f"8K dense causal block sweep (npu_fa baseline = {t_npu:.2f} ms)")
        print("-" * 72)
        print(f"{'BLOCK_M':<10}{'BLOCK_N':<10}{'triton(ms)':<14}{'ratio':<10}{'note':<20}")
        print("-" * 72)

        for bm, bn in block_pairs:
            def f(q, k, v, scale, _bm=bm, _bn=bn):
                return triton_ascend_fa_dense(
                    q, k, v, sm_scale=scale, causal=True,
                    block_m=_bm, block_n=_bn,
                )
            try:
                t = _bench_one(f, q, k, v, scale, warmup=2, iters=5)
                ratio = t / t_npu
                print(f"{bm:<10}{bn:<10}{t:<14.2f}{ratio:<10.2f}")
            except Exception as e:  # noqa: BLE001
                short = f"{e.__class__.__name__}: {str(e)[:60]}"
                print(f"{bm:<10}{bn:<10}{'FAIL':<14}{'--':<10}{short:<20}")

        print("-" * 72)
        print("解读：找出 ratio 最小的 (BLOCK_M, BLOCK_N) — 多数 NPU FA kernel 实践")
        print("      最优 tile 在 (128,64) / (128,128) / (256,128) 附近。")
    except Exception as e:  # noqa: BLE001
        print(f"[sweep] FATAL: {e.__class__.__name__}: {e}")
        if verbose:
            traceback.print_exc()


def run_benchmark(verbose: bool = False) -> None:
    """8K / 16K dense causal benchmark — triton-ascend vs npu_fusion_attention。"""
    try:
        from minference.ops.triton_ascend_fa_poc import (
            has_triton,
            triton_ascend_fa_dense,
        )

        if not has_triton():
            print("[bench] SKIP — triton-ascend 不可用")
            return

        dev = _require_npu(verbose)

        # Llama-3 8B prefill 近似：head_num=32 用 4 (单卡跑 8B 时 PoC 简化 — 主要看 S 维度 scaling)
        # D=128 是 Llama-3 head_dim
        configs = [
            ("8K", 1, 4, 8192, 128),
            ("16K", 1, 4, 16384, 128),
        ]

        print("-" * 72)
        print(f"{'config':<10}{'shape':<28}{'triton(ms)':<14}{'npu_fa(ms)':<14}{'ratio':<10}")
        print("-" * 72)

        torch.manual_seed(0)
        for tag, B, N, S, D in configs:
            q = torch.randn(B, N, S, D, dtype=torch.float16, device=dev)
            k = torch.randn(B, N, S, D, dtype=torch.float16, device=dev)
            v = torch.randn(B, N, S, D, dtype=torch.float16, device=dev)
            scale = 1.0 / math.sqrt(D)

            def f_triton(q, k, v, scale):
                return triton_ascend_fa_dense(q, k, v, sm_scale=scale, causal=True)

            def f_npu(q, k, v, scale):
                return _npu_fa_dense_causal(q, k, v, scale)

            try:
                t_triton = _bench_one(f_triton, q, k, v, scale)
            except Exception as e:  # noqa: BLE001
                print(f"  [{tag}] triton FAIL: {e.__class__.__name__}: {e}")
                continue
            try:
                t_npu = _bench_one(f_npu, q, k, v, scale)
            except Exception as e:  # noqa: BLE001
                print(f"  [{tag}] npu_fa FAIL: {e.__class__.__name__}: {e}")
                continue

            ratio = t_triton / t_npu if t_npu > 0 else float("inf")
            shape_s = f"[{B},{N},{S},{D}]"
            print(f"{tag:<10}{shape_s:<28}{t_triton:<14.2f}{t_npu:<14.2f}{ratio:<10.2f}")

        print("-" * 72)
        print("解读：ratio <= 3.0 表示 triton-ascend PoC 在合理量级（high-level DSL 上限），")
        print("      可启动 PR-4-VS；ratio > 5.0 提示 kernel 配置或 triton-ascend 不成熟需诊断。")

    except Exception as e:  # noqa: BLE001
        print(f"[bench] FATAL: {e.__class__.__name__}: {e}")
        if verbose:
            traceback.print_exc()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="PR-4-poc：triton-ascend dense FA vs npu_fusion_attention")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--skip-bench", action="store_true", help="只跑精度，不跑 benchmark")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="对 (BLOCK_M, BLOCK_N) 做 sweep（诊断 ratio 异常时用，跑完后退出）",
    )
    args = parser.parse_args()

    if args.sweep:
        print("=" * 72)
        print("PR-4-poc BLOCK sweep（诊断模式）")
        print("=" * 72)
        run_block_sweep(args.verbose)
        return 0

    print("=" * 72)
    print("PR-4-poc：triton-ascend dense FA kernel 验收（精度 + 8K/16K 性能）")
    print("=" * 72)

    precision_ok = test_precision(args.verbose)

    if not args.skip_bench:
        print()
        print("=" * 72)
        print("Benchmark (8K / 16K dense causal)")
        print("=" * 72)
        run_benchmark(args.verbose)

    print()
    if precision_ok:
        print("精度 PASS — PR-4-poc 精度 gate 通过")
        return 0
    print("精度 FAIL — PR-4-poc 阻塞，先修 kernel 再 bench")
    return 1


if __name__ == "__main__":
    sys.exit(main())
