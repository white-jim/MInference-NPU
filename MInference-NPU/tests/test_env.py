# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""M0 环境烟测。

默认只跑当前 v1 必需的 torch_npu / npu_fusion_attention 烟测。

可选子测试：
1. test_triton_ascend_vector_add（--with-triton）
   跑最小 Triton-Ascend vector add kernel，与 PyTorch CPU 参考逐元素对比，
   验证 triton-ascend 工具链 / JIT 编译 / kernel launch 链路。

必跑子测试：
2. test_npu_fusion_attention_smoke
   调 torch_npu.npu_fusion_attention 跑一个小尺寸 dense causal attention，
   与手写 PyTorch eager（softmax + causal mask）参考对比，
   验证 dense FA API 可用，为 M1 全 dense 链路铺路。

跑法：
    python tests/test_env.py            # 标准模式，print PASS/FAIL
    python tests/test_env.py -v         # 详细输出（含 shape / 差异）

退出码：已启用的测试全部 PASS 退出 0，任一 FAIL 退出 1。

注意：必须在 Linux + Ascend 驱动 + CANN + torch_npu 的目标 NPU 机器上跑。
在 Windows 工作机上 import torch_npu 会失败，脚本会清晰报错并 FAIL。
"""

import argparse
import sys
import traceback

import torch


# ----------------------------------------------------------------------------
# 通用小工具
# ----------------------------------------------------------------------------


def _print_status(name: str, ok: bool, msg: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if msg:
        line += f"  ({msg})"
    print(line)


def _require_npu(verbose: bool) -> "torch.device":
    """加载 torch_npu，返回 npu:0；失败抛 RuntimeError 给上层捕获。"""
    try:
        import torch_npu  # noqa: F401 — 触发 NPU 后端注册
    except ImportError as e:
        raise RuntimeError(
            "torch_npu import failed — 当前环境不是昇腾 NPU 机器，或 CANN 未 source。"
            f" 原始错误：{e}"
        )
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() == False — 驱动 / 固件未就绪")
    dev = torch.device("npu:0")
    if verbose:
        print(
            f"    [env] torch={torch.__version__} "
            f"npu_count={torch.npu.device_count()} "
            f"device={dev}"
        )
    return dev


# ----------------------------------------------------------------------------
# 子测试 1：Triton-Ascend vector add
# ----------------------------------------------------------------------------


def test_triton_ascend_vector_add(verbose: bool = False) -> bool:
    name = "test_triton_ascend_vector_add"
    try:
        dev = _require_npu(verbose)

        # triton-ascend 与 CUDA Triton 同名 import，由 torch_npu 注册后端切换
        import triton
        import triton.language as tl

        @triton.jit
        def _vec_add_kernel(
            x_ptr,
            y_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        N = 8192
        BLOCK = 1024
        torch.manual_seed(0)
        x_cpu = torch.randn(N, dtype=torch.float32)
        y_cpu = torch.randn(N, dtype=torch.float32)
        ref = x_cpu + y_cpu

        x = x_cpu.to(dev)
        y = y_cpu.to(dev)
        out = torch.empty_like(x)

        grid = ((N + BLOCK - 1) // BLOCK,)
        _vec_add_kernel[grid](x, y, out, N, BLOCK_SIZE=BLOCK)

        # 跨 device 比对：把 NPU 结果拉回 CPU
        out_cpu = out.cpu()
        max_abs = (out_cpu - ref).abs().max().item()
        ok = torch.allclose(out_cpu, ref, atol=1e-5, rtol=1e-5)

        if verbose:
            print(f"    [vec_add] N={N} BLOCK={BLOCK} max_abs_diff={max_abs:.3e}")
        _print_status(name, ok, f"max_abs_diff={max_abs:.3e}")
        return ok

    except Exception as e:  # noqa: BLE001 — 烟测要捕获所有异常并继续跑下一个子测试
        _print_status(name, False, f"exception: {e.__class__.__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


# ----------------------------------------------------------------------------
# 子测试 2：npu_fusion_attention smoke（dense causal）
# ----------------------------------------------------------------------------


def _eager_attention_ref(q, k, v, scale, causal: bool):
    """PyTorch eager 参考实现（fp32 内算，输出 cast 回输入 dtype）。

    输入形状：[B, N, S, D]  与 npu_fusion_attention 的 BNSD 对齐。
    """
    in_dtype = q.dtype
    qf = q.float()
    kf = k.float()
    vf = v.float()
    attn = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    if causal:
        S_q = qf.shape[-2]
        S_k = kf.shape[-2]
        mask = torch.ones(S_q, S_k, device=qf.device, dtype=torch.bool).tril()
        attn = attn.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(attn, dim=-1)
    out = torch.matmul(probs, vf)
    return out.to(in_dtype)


def test_npu_fusion_attention_smoke(verbose: bool = False) -> bool:
    name = "test_npu_fusion_attention_smoke"
    try:
        dev = _require_npu(verbose)
        import torch_npu  # noqa: F401  — API 通过 torch_npu.npu_fusion_attention 调用
        import math

        # 小尺寸：1 batch / 4 head / 256 ctx / head_dim 128（2 的幂、bf16 友好）
        B, N, S, D = 1, 4, 256, 128
        dtype = torch.float16  # 910B 上 fp16 / bf16 都通，fp16 数值噪声范围熟悉一些
        scale = 1.0 / math.sqrt(D)

        torch.manual_seed(0)
        q = torch.randn(B, N, S, D, dtype=dtype, device=dev)
        k = torch.randn(B, N, S, D, dtype=dtype, device=dev)
        v = torch.randn(B, N, S, D, dtype=dtype, device=dev)

        # npu_fusion_attention 接口（BNSD layout，sparse_mode=2 表示 causal）
        # 参考 torch_npu 文档：返回 (attention_out, softmax_max, softmax_sum, ...)
        # 不同 torch_npu 小版本返回元组长度可能略有差异，只取第一个即可
        result = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=N,
            input_layout="BNSD",
            scale=scale,
            sparse_mode=2,  # 2 = causal triangular
        )
        out_npu = result[0] if isinstance(result, (tuple, list)) else result

        # 参考输出在 NPU 上算（fp32 中间，避免比对时的设备一致性问题）
        with torch.no_grad():
            out_ref = _eager_attention_ref(q, k, v, scale, causal=True)

        diff = (out_npu.float() - out_ref.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        ok = max_abs < 1e-2  # fp16 + softmax 累积噪声范围

        if verbose:
            print(
                f"    [npu_fa] shape={tuple(out_npu.shape)} dtype={out_npu.dtype} "
                f"max_abs_diff={max_abs:.3e} mean_abs_diff={mean_abs:.3e}"
            )
        _print_status(name, ok, f"max_abs_diff={max_abs:.3e}")
        return ok

    except Exception as e:  # noqa: BLE001
        _print_status(name, False, f"exception: {e.__class__.__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="M0 NPU 环境烟测")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument(
        "--with-triton",
        action="store_true",
        help="额外验证 Triton-Ascend。CANN 8.1.RC1 默认不要求安装。",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MInference-NPU M0 环境烟测")
    print("=" * 60)

    results = [test_npu_fusion_attention_smoke(args.verbose)]
    if args.with_triton:
        results.insert(0, test_triton_ascend_vector_add(args.verbose))

    print("-" * 60)
    if all(results):
        print("ALL PASS — M0 环境就绪，可推进 M1")
        return 0
    n_fail = sum(1 for r in results if not r)
    print(f"{n_fail}/{len(results)} FAIL — 请按 docs/SETUP.md §5 常见踩坑排查")
    return 1


if __name__ == "__main__":
    sys.exit(main())
