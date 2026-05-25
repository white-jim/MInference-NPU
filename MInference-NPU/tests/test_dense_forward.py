# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""M1 端到端 dense fallback 测试。

验证：
1. `MInference(attn_type="dense")(model)` patch 之后能跑完整 forward 而不挂
2. patch 后的输出与 HF 原生 eager attention 输出差异 < 1e-2（NPU 数值噪声范围内）
3. accelerate 多卡（device_map="auto"）切层后仍能正确跑（要求 ≥ 2 张 NPU；单卡环境跳过）

不依赖外网模型权重。使用一个 tiny LlamaConfig 现场构造小模型，避免拉 ~14GB 的真权重。

跑法：
    python tests/test_dense_forward.py            # 标准模式
    python tests/test_dense_forward.py -v         # 详细输出

退出码：全 PASS → 0；任一 FAIL → 1。
"""

from __future__ import annotations

import argparse
import sys
import traceback


def _print_status(name: str, ok: bool, msg: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if msg:
        line += f"  ({msg})"
    print(line)


def _build_tiny_llama(device: str, dtype):
    """构造一个 4-layer / 4-head / hidden=256 的 Llama，避免拉远端权重。"""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        rope_theta=10000.0,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(device).to(dtype)
    model.eval()
    return model, cfg


def _require_npu(verbose: bool):
    """加载 torch_npu。失败抛 RuntimeError 给上层处理。返回 device 字符串。"""
    import torch

    try:
        import torch_npu  # noqa: F401
    except ImportError as e:
        raise RuntimeError(f"torch_npu import 失败：{e}") from e
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() False")
    if verbose:
        print(
            f"    [env] torch={torch.__version__} "
            f"npu_count={torch.npu.device_count()}"
        )
    return "npu:0"


# ----------------------------------------------------------------------------
# 子测试 1: dense fallback 数值正确性
# ----------------------------------------------------------------------------


def test_dense_fallback_matches_eager(verbose: bool = False) -> bool:
    name = "test_dense_fallback_matches_eager"
    try:
        import torch

        device = _require_npu(verbose)
        dtype = torch.float16

        # 基线：HF 原生 eager attention
        model_a, cfg = _build_tiny_llama(device, dtype)
        torch.manual_seed(42)
        input_ids = torch.randint(0, cfg.vocab_size, (1, 256), device=device)

        with torch.no_grad():
            out_eager = model_a(input_ids).logits

        # MInference dense fallback
        from minference import MInference

        model_b, _ = _build_tiny_llama(device, dtype)
        # 确保两个模型权重一致（同 seed 已保证）
        model_b.load_state_dict(model_a.state_dict())
        model_b = MInference(attn_type="dense")(model_b)

        with torch.no_grad():
            out_npu = model_b(input_ids).logits

        diff = (out_eager.float() - out_npu.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        ok = max_abs < 1e-2

        if verbose:
            print(
                f"    [dense] shape={tuple(out_npu.shape)} "
                f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}"
            )
        _print_status(name, ok, f"max_abs={max_abs:.3e}")
        return ok

    except Exception as e:  # noqa: BLE001
        _print_status(name, False, f"{e.__class__.__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


# ----------------------------------------------------------------------------
# 子测试 2: minference attn_type 走完整 per-head 调度路径
# ----------------------------------------------------------------------------


def test_minference_perhead_dense_fallback(verbose: bool = False) -> bool:
    name = "test_minference_perhead_dense_fallback"
    try:
        import json
        import os
        import tempfile

        import torch

        device = _require_npu(verbose)
        dtype = torch.float16

        # 构造一个临时 best_pattern JSON（4 层，每层 4 head 都是 vertical_and_slash）
        # 让 minference 走 per-head 路径而不是 attn_type="dense" 的快速路
        best_pattern = [
            {str(h): ["vertical_and_slash", 500, 1024, 0.9] for h in range(4)}
            for _ in range(4)
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(best_pattern, f)
            config_path = f.name

        try:
            from minference import MInference

            model, _ = _build_tiny_llama(device, dtype)
            model = MInference(
                attn_type="minference",
                config_path=config_path,
                starting_layer=0,
            )(model)

            torch.manual_seed(7)
            input_ids = torch.randint(0, 1024, (1, 512), device=device)
            with torch.no_grad():
                out = model(input_ids).logits

            ok = (
                not torch.isnan(out).any().item()
                and not torch.isinf(out).any().item()
                and tuple(out.shape) == (1, 512, 1024)
            )
            if verbose:
                print(
                    f"    [perhead] shape={tuple(out.shape)} "
                    f"has_nan={torch.isnan(out).any().item()} "
                    f"has_inf={torch.isinf(out).any().item()}"
                )
            _print_status(name, ok)
            return ok
        finally:
            os.unlink(config_path)

    except Exception as e:  # noqa: BLE001
        _print_status(name, False, f"{e.__class__.__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


# ----------------------------------------------------------------------------
# 子测试 3: accelerate 多卡 device_map="auto"（≥ 2 张 NPU 时跑，否则 SKIP）
# ----------------------------------------------------------------------------


def test_accelerate_device_map_auto(verbose: bool = False) -> bool | str:
    name = "test_accelerate_device_map_auto"
    try:
        import torch

        _require_npu(verbose)
        if torch.npu.device_count() < 2:
            _print_status(name, True, "SKIP — 单卡环境，跳过多卡测试")
            return True  # SKIP 视为 PASS（环境限制）

        from accelerate import dispatch_model, infer_auto_device_map

        from minference import MInference

        # 这里复用 _build_tiny_llama，但 device 先放 cpu，让 accelerate 切到 2 张 NPU 上
        model, cfg = _build_tiny_llama("cpu", torch.float16)
        device_map = infer_auto_device_map(
            model, no_split_module_classes=["LlamaDecoderLayer"]
        )
        # device_map 默认会把不同 layer 分到 npu:0/npu:1
        model = dispatch_model(model, device_map=device_map)

        model = MInference(attn_type="dense")(model)

        torch.manual_seed(0)
        # accelerate dispatch_model 在 NPU 上未稳定地把 CPU 输入自动搬到 embed 卡（hook
        # 触发依赖 torch 后端的 device 识别），显式 .to(embed_device) 避开此问题；本测试
        # 关注的是多卡切层后跨设备 forward 是否能跑通，而不是 accelerate-on-NPU 的 hook 行为。
        embed_device = next(model.model.embed_tokens.parameters()).device
        input_ids = torch.randint(
            0, cfg.vocab_size, (1, 128), device=embed_device
        )
        with torch.no_grad():
            out = model(input_ids).logits

        ok = (
            not torch.isnan(out).any().item()
            and tuple(out.shape) == (1, 128, cfg.vocab_size)
        )
        if verbose:
            print(f"    [multinpu] device_map={device_map}")
        _print_status(name, ok)
        return ok

    except Exception as e:  # noqa: BLE001
        _print_status(name, False, f"{e.__class__.__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 dense fallback 端到端测试")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("MInference-NPU M1 — Dense Fallback 端到端测试")
    print("=" * 60)

    results = [
        test_dense_fallback_matches_eager(args.verbose),
        test_minference_perhead_dense_fallback(args.verbose),
        test_accelerate_device_map_auto(args.verbose),
    ]

    print("-" * 60)
    if all(results):
        print("ALL PASS — M1 dense fallback 链路就绪，可推进 M2")
        return 0
    n_fail = sum(1 for r in results if not r)
    print(f"{n_fail}/{len(results)} FAIL — 见 docs/M1_dense_pipeline.md §排查")
    return 1


if __name__ == "__main__":
    sys.exit(main())
