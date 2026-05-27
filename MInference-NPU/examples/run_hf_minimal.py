# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""Phi-3 HF smoke runner for the current PR-4 TileLang path-B work.

用法：
    # 默认跑本地 Phi-3-mini-128k-instruct，单卡 npu:0
    python examples/run_hf_minimal.py

    # 指定 path-B probe config / 长度
    python examples/run_hf_minimal.py \
        --config-path minference/configs/Phi_3_mini_128k_instruct_pathb_stream_llm_aligned_dense_others.json \
        --ctx-len 4096 \
        --profile-branches --num-runs 2

注意：
- 当前默认服务 Phi-3 path-B 调试。其他模型配置已从精简工作区移除。
- 速度必须和 `--attn-type dense` baseline 对比看。
"""

from __future__ import annotations

import argparse
import os
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="MInference-NPU 最小 HF 示例")
    parser.add_argument(
        "--model",
        default="microsoft/Phi-3-mini-128k-instruct",
        help="HF 模型名。当前精简工作区只保留 Phi-3 128K config。",
    )
    parser.add_argument(
        "--model-path",
        default="/data/guoshiyao/resources/models/Phi-3-mini-128k-instruct",
        help="可选：本地权重目录。给定时 from_pretrained 走该路径，"
             "best_pattern 仍按 --model 在 MODEL2PATH 里查。",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="可选：显式 best_pattern JSON。用于 Phi3 path-B probe 等临时配置。",
    )
    parser.add_argument(
        "--ctx-len",
        type=int,
        default=8192,
        help="prompt 长度（token 数）。首次跑建议 8k，确认能跑通后再加",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="解码 token 数；当前主要用 1-16 做 smoke/profile",
    )
    parser.add_argument(
        "--device-map",
        default="npu:0",
        help='"npu:0" 单卡 / "auto" accelerate 自动多卡 / 具体 device_map dict（JSON 串）',
    )
    parser.add_argument(
        "--attn-type",
        default="minference",
        choices=("minference", "dense", "hf"),
        help='"minference" 走 per-head 调度 / "dense" 全部 dense / "hf" 不 patch',
    )
    parser.add_argument(
        "--profile-branches",
        action="store_true",
        help="可选：同步计时 minference_forward 内 dense/VS/stream/block 分支，用于定位端到端瓶颈。",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="同一进程内重复 generate 次数；用于区分首次 TileLang JIT 与 steady-state。",
    )
    args = parser.parse_args()

    hf_cache = "/data/guoshiyao/resources/.hf_cache"
    os.environ.setdefault("HF_HOME", hf_cache)
    os.makedirs(hf_cache, exist_ok=True)

    import torch

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("[ERROR] torch_npu 未安装，本示例必须在昇腾 NPU 机器上跑")
        return 1

    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:  # pragma: no cover - transformers 旧版本没有该入口
        DynamicCache = None

    from minference import MInference

    # Phi3 remote modeling in older snapshots still reads ``seen_tokens``.
    # transformers 4.57 DynamicCache exposes ``get_seq_length()`` instead.
    if DynamicCache is not None and not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())  # type: ignore[attr-defined]
    if DynamicCache is not None and not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: self.get_max_cache_shape()  # type: ignore[attr-defined]
    if DynamicCache is not None and not hasattr(DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length, layer_idx=0):
            max_length = self.get_max_length()
            previous_seq_length = self.get_seq_length(layer_idx)
            if max_length is not None and previous_seq_length + new_seq_length > max_length:
                return max_length - new_seq_length
            return previous_seq_length

        DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]

    load_src = args.model_path or args.model
    print(f"[1/4] 加载 tokenizer & model: {load_src}"
          + (f"  (best_pattern key={args.model})" if args.model_path else ""))
    tokenizer = AutoTokenizer.from_pretrained(load_src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        load_src,
        torch_dtype=torch.float16,
        device_map=args.device_map,
        trust_remote_code=True,
        attn_implementation="eager",  # NPU 上没 flash_attn，强制 eager；patch 后会替换
    )
    model.eval()

    if args.attn_type != "hf":
        print(f"[2/4] 应用 MInference patch (attn_type={args.attn_type})")
        model = MInference(
            attn_type=args.attn_type,
            model_name=(
                args.model
                if args.attn_type == "minference" and not args.config_path
                else None
            ),
            config_path=args.config_path,
        )(model)
    else:
        print("[2/4] 跳过 patch (attn_type='hf')")

    pathb_counts = {}
    pathb_restores = []
    branch_stats = {}
    branch_restores = []

    def _sync_npu():
        if hasattr(torch, "npu"):
            torch.npu.synchronize()

    def _wrap_timed(module, attr, key, restores, stats=None, count_success=False):
        original = getattr(module, attr)
        target_stats = branch_stats if stats is None else stats
        target_stats.setdefault(key, {"count": 0, "seconds": 0.0})

        def wrapped(*w_args, **w_kwargs):
            _sync_npu()
            t0 = time.perf_counter()
            out = original(*w_args, **w_kwargs)
            _sync_npu()
            target_stats[key]["seconds"] += time.perf_counter() - t0
            target_stats[key]["count"] += 1
            if count_success:
                pathb_counts[key] = pathb_counts.get(key, 0) + 1
            return out

        setattr(module, attr, wrapped)
        restores.append((module, attr, original))

    if args.attn_type == "minference":
        try:
            import minference.ops.block_sparse_kernel_npu as block_sparse_kernel_npu
            import minference.ops.streaming_kernel_npu as streaming_kernel_npu

            _wrap_timed(
                block_sparse_kernel_npu,
                "_block_sparse_tilelang_npu",
                "block_sparse",
                pathb_restores,
                stats={},
                count_success=True,
            )
            _wrap_timed(
                streaming_kernel_npu,
                "_streaming_tilelang_npu",
                "stream_llm",
                pathb_restores,
                stats={},
                count_success=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    path-B 计数器安装失败：{exc}")

    if args.profile_branches and args.attn_type == "minference":
        try:
            import minference.modules.minference_forward as minference_forward

            _wrap_timed(minference_forward, "dense_attention", "dense", branch_restores)
            _wrap_timed(minference_forward, "_streaming_forward", "stream_llm", branch_restores)
            _wrap_timed(
                minference_forward,
                "_block_sparse_attention",
                "block_sparse",
                branch_restores,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    branch profiler 安装失败：{exc}")

    print(f"[3/4] 构造长 prompt（ctx_len={args.ctx_len}）")
    # 构造一个简单的可重复 prompt：把一句话循环到目标 token 数
    base = "The quick brown fox jumps over the lazy dog. " * 200
    enc = tokenizer(base, return_tensors="pt", truncation=True, max_length=args.ctx_len)
    # 不足 ctx_len 就再拼，直到达到目标长度
    while enc["input_ids"].shape[1] < args.ctx_len:
        more = tokenizer(base, return_tensors="pt", truncation=True,
                         max_length=args.ctx_len - enc["input_ids"].shape[1])
        enc["input_ids"] = torch.cat([enc["input_ids"], more["input_ids"]], dim=1)
        enc["attention_mask"] = torch.cat(
            [enc["attention_mask"], more["attention_mask"]], dim=1
        )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    print(f"    实际 prompt 长度：{input_ids.shape[1]} tokens")

    print(f"[4/4] generate(max_new_tokens={args.max_new_tokens})")
    if args.num_runs < 1:
        raise ValueError("--num-runs must be >= 1")
    run_times = []
    out = None
    try:
        with torch.no_grad():
            model_device = next(model.parameters()).device
            input_ids_device = input_ids.to(model_device)
            attention_mask_device = attention_mask.to(model_device)
            for run_idx in range(args.num_runs):
                t0 = time.time()
                out = model.generate(
                    input_ids_device,
                    attention_mask=attention_mask_device,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
                dt_run = time.time() - t0
                run_times.append(dt_run)
                print(f"    run {run_idx + 1}/{args.num_runs}: {dt_run:.2f}s")
    finally:
        for module, attr, original in pathb_restores:
            setattr(module, attr, original)
        for module, attr, original in branch_restores:
            setattr(module, attr, original)
    assert out is not None
    dt = sum(run_times)
    new_tokens = out[0, input_ids.shape[1]:]
    print(f"    完成，累计用时 {dt:.2f}s，解码 {len(new_tokens)} tokens")
    if pathb_counts:
        print(
            "    path-B hits: "
            + ", ".join(f"{name}={count}" for name, count in pathb_counts.items())
        )
    if branch_stats:
        print("    branch timings:")
        for name, item in sorted(branch_stats.items()):
            print(f"      {name}: {item['seconds']:.3f}s over {item['count']} calls")
    print(f"    输出文本：{tokenizer.decode(new_tokens, skip_special_tokens=True)!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
