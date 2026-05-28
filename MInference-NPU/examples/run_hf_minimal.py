# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""Phi-3 HF smoke runner for the current PR-4 sparse-attention work.

用法：
    # 默认跑本地 Phi-3-mini-128k-instruct，单卡 npu:0
    python examples/run_hf_minimal.py

    # 指定 sparse probe config / 长度
    python examples/run_hf_minimal.py \
        --config-path minference/configs/Phi_3_mini_128k_instruct_pathb_stream_llm_aligned_dense_others.json \
        --ctx-len 4096 \
        --profile-branches --num-runs 2

注意：
- 当前默认服务 Phi-3 sparse attention 调试。其他模型配置已从精简工作区移除。
- 速度必须和 `--attn-type dense` baseline 对比看。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "benchmarks" / "results" / "runs"


class _Tee:
    """Minimal stdout tee that forwards writes to multiple text streams."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self._streams:
            st.flush()

    def isatty(self):  # generate() / tqdm 偶尔会查询；保持 False，无 fancy 控制字符
        return False


def _resolve_run_path(spec: Path | None) -> Path | None:
    """Bare-filename → benchmarks/results/runs/<filename>；其他保持原样。

    设计目的：让 ``--save-output dense_32k.json`` 自动落到 runs 目录，
    同时保留对 ``./xx`` / 绝对路径 / 多段相对路径的尊重。
    """
    if spec is None:
        return None
    if not spec.is_absolute() and len(spec.parts) == 1:
        return DEFAULT_RUN_DIR / spec.name
    return spec


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
        help="可选：显式 best_pattern JSON。用于 Phi3 sparse probe 等临时配置。",
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
        help="同一进程内重复 generate 次数；用于区分首次开销/JIT 与 steady-state。",
    )
    parser.add_argument(
        "--empty-cache-between-runs",
        action="store_true",
        help="长上下文调试用：每次 generate 后释放输出并 empty_cache，避免 HF 4D mask 重复分配导致 OOM。",
    )
    parser.add_argument(
        "--save-output",
        type=Path,
        default=None,
        help="持久化 generate 输出（token ids + per-step top-K logits）到 JSON，"
             "供后续 --compare-to 做 sparse vs dense 质量对照。"
             "强制 generate(output_scores=True, return_dict_in_generate=True)。",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="读取另一份 --save-output 生成的 JSON 与当前 run 对比，"
             "输出 token 匹配率、top-1/top-5 命中、KL divergence（基于 union top-K）。",
    )
    parser.add_argument(
        "--quality-top-k",
        type=int,
        default=32,
        help="--save-output 时每步保留的 top-K 个 token / logits，"
             "用于 compare 阶段近似 KL。默认 32，足够覆盖常见 top-5/10 重合 + KL 近似。",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="把脚本 stdout tee 到该文件。裸文件名会被放到 "
             f"{DEFAULT_RUN_DIR.relative_to(REPO_ROOT)} 下，便于 commit/push 后在开发机 git pull 查看。",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="便捷参数：等价于 --log-file <NAME>.log（除非 --log-file 显式给定）。",
    )
    args = parser.parse_args()

    # --- 解析输出路径：裸文件名 → benchmarks/results/runs/<file> ---
    args.save_output = _resolve_run_path(args.save_output)
    args.compare_to = _resolve_run_path(args.compare_to)
    args.log_file = _resolve_run_path(args.log_file)
    if args.log_file is None and args.run_name:
        args.log_file = DEFAULT_RUN_DIR / f"{args.run_name}.log"

    # --- 安装 stdout tee（如有） ---
    log_handle = None
    orig_stdout = sys.stdout
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = _Tee(orig_stdout, log_handle)
        print(f"[log] tee stdout -> {args.log_file}")

    try:
        return _run_with_args(args)
    finally:
        if log_handle is not None:
            sys.stdout = orig_stdout
            log_handle.close()


def _run_with_args(args: argparse.Namespace) -> int:

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
            # stream_llm 已切到 hardware band+sink；命中计数挂在 _streaming_npu 上。
            _wrap_timed(
                streaming_kernel_npu,
                "_streaming_npu",
                "stream_llm",
                pathb_restores,
                stats={},
                count_success=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    sparse path 计数器安装失败：{exc}")

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
    if args.save_output is not None and args.quality_top_k < 1:
        raise ValueError("--quality-top-k must be >= 1 when --save-output is given")
    run_times = []
    out = None
    last_scores: list | None = None
    want_scores = args.save_output is not None or args.compare_to is not None
    try:
        with torch.no_grad():
            model_device = next(model.parameters()).device
            input_ids_device = input_ids.to(model_device)
            attention_mask_device = attention_mask.to(model_device)
            for run_idx in range(args.num_runs):
                t0 = time.time()
                gen_kwargs = dict(
                    attention_mask=attention_mask_device,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
                if want_scores:
                    gen_kwargs["output_scores"] = True
                    gen_kwargs["return_dict_in_generate"] = True
                gen_out = model.generate(input_ids_device, **gen_kwargs)
                if want_scores:
                    out = gen_out.sequences
                    last_scores = list(gen_out.scores)
                else:
                    out = gen_out
                dt_run = time.time() - t0
                run_times.append(dt_run)
                print(f"    run {run_idx + 1}/{args.num_runs}: {dt_run:.2f}s")
                if args.empty_cache_between_runs and run_idx + 1 < args.num_runs:
                    del out, gen_out
                    out = None
                    if want_scores:
                        last_scores = None
                    _sync_npu()
                    gc.collect()
                    if hasattr(torch, "npu"):
                        torch.npu.empty_cache()
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
            "    sparse path hits: "
            + ", ".join(f"{name}={count}" for name, count in pathb_counts.items())
        )
    if branch_stats:
        print("    branch timings:")
        for name, item in sorted(branch_stats.items()):
            print(f"      {name}: {item['seconds']:.3f}s over {item['count']} calls")
    print(f"    输出文本：{tokenizer.decode(new_tokens, skip_special_tokens=True)!r}")

    if args.save_output is not None:
        assert last_scores is not None, "internal error: --save-output but no scores captured"
        _save_quality_output(
            path=args.save_output,
            generated_token_ids=new_tokens.tolist(),
            scores=last_scores,
            top_k=args.quality_top_k,
            metadata={
                "model": args.model,
                "model_path": args.model_path,
                "config_path": str(args.config_path) if args.config_path else None,
                "ctx_len": int(input_ids.shape[1]),
                "max_new_tokens": int(args.max_new_tokens),
                "attn_type": args.attn_type,
                "device_map": args.device_map,
                "quality_top_k": int(args.quality_top_k),
                "run_times_seconds": [float(t) for t in run_times],
            },
        )
        print(f"    [quality] 保存输出 -> {args.save_output}")

    if args.compare_to is not None:
        assert last_scores is not None, "--compare-to requires --save-output to capture scores"
        _compare_quality_outputs(
            reference_path=args.compare_to,
            current_token_ids=new_tokens.tolist(),
            current_scores=last_scores,
            top_k=args.quality_top_k,
        )

    return 0


def _save_quality_output(
    path: Path,
    generated_token_ids: list[int],
    scores,
    top_k: int,
    metadata: dict,
) -> None:
    """Dump greedy decode token ids + per-step top-K logits to JSON."""
    import torch as _torch

    per_step = []
    for step_logits in scores:
        # step_logits: [batch=1, vocab]; greedy, only batch entry 0.
        v = step_logits[0].detach().float().cpu()
        k = min(top_k, v.shape[-1])
        topv, topi = _torch.topk(v, k=k)
        per_step.append(
            {
                "token_ids": topi.tolist(),
                "logits": topv.tolist(),
            }
        )
    payload = {
        "metadata": metadata,
        "generated_token_ids": [int(t) for t in generated_token_ids],
        "per_step_top_k": per_step,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f)
        f.write("\n")


def _compare_quality_outputs(
    reference_path: Path,
    current_token_ids: list[int],
    current_scores,
    top_k: int,
) -> None:
    import torch as _torch

    with reference_path.open() as f:
        ref = json.load(f)
    ref_meta = ref.get("metadata", {})
    ref_tokens = ref["generated_token_ids"]
    ref_per_step = ref["per_step_top_k"]

    cur_tokens = [int(t) for t in current_token_ids]
    cur_per_step = []
    for step_logits in current_scores:
        v = step_logits[0].detach().float().cpu()
        k = min(top_k, v.shape[-1])
        topv, topi = _torch.topk(v, k=k)
        cur_per_step.append(
            {
                "token_ids": topi.tolist(),
                "logits": topv.tolist(),
            }
        )

    n_compare = min(len(ref_tokens), len(cur_tokens), len(ref_per_step), len(cur_per_step))
    print()
    print("    [quality] reference:", reference_path)
    print(
        f"    [quality] reference attn_type={ref_meta.get('attn_type')!r}, "
        f"config_path={ref_meta.get('config_path')!r}, "
        f"ctx_len={ref_meta.get('ctx_len')}, "
        f"max_new_tokens={ref_meta.get('max_new_tokens')}, "
        f"top_k={ref_meta.get('quality_top_k')}"
    )
    print(
        f"    [quality] compare first {n_compare} steps "
        f"(ref_len={len(ref_tokens)}, cur_len={len(cur_tokens)})"
    )

    if n_compare == 0:
        print("    [quality] nothing to compare.")
        return

    # ---- 1. token sequence match ----
    seq_match_total = 0
    first_div = None
    longest_prefix = 0
    streak = True
    for i in range(n_compare):
        if cur_tokens[i] == ref_tokens[i]:
            seq_match_total += 1
            if streak:
                longest_prefix += 1
        else:
            if streak:
                first_div = i
                streak = False
    print(
        f"    [quality] token greedy match: "
        f"{seq_match_total}/{n_compare} "
        f"({100.0 * seq_match_total / n_compare:.1f}%); "
        f"longest matching prefix = {longest_prefix}; "
        f"first divergence step = {first_div}"
    )

    # ---- 2. top-1 / top-5 set agreement (ref-vs-cur per step) ----
    top1_match = 0
    top5_overlap_sum = 0.0
    for i in range(n_compare):
        ref_ids = ref_per_step[i]["token_ids"]
        cur_ids = cur_per_step[i]["token_ids"]
        if not ref_ids or not cur_ids:
            continue
        if ref_ids[0] == cur_ids[0]:
            top1_match += 1
        k5 = min(5, len(ref_ids), len(cur_ids))
        ref_set = set(ref_ids[:k5])
        cur_set = set(cur_ids[:k5])
        if ref_set and cur_set:
            top5_overlap_sum += len(ref_set & cur_set) / float(k5)
    print(
        f"    [quality] per-step top-1 match: "
        f"{top1_match}/{n_compare} "
        f"({100.0 * top1_match / n_compare:.1f}%); "
        f"per-step top-5 jaccard mean = "
        f"{top5_overlap_sum / n_compare:.3f}"
    )

    # ---- 3. KL divergence on union of top-K (approx) ----
    # Reference -> current direction: KL(P_ref || Q_cur) using union(top-K) per step.
    # Missing tokens use logit = -inf -> prob 0; safe because we take union.
    kl_sum = 0.0
    kl_n = 0
    for i in range(n_compare):
        ref_ids = ref_per_step[i]["token_ids"]
        ref_logits = ref_per_step[i]["logits"]
        cur_ids = cur_per_step[i]["token_ids"]
        cur_logits = cur_per_step[i]["logits"]
        ref_map = dict(zip(ref_ids, ref_logits))
        cur_map = dict(zip(cur_ids, cur_logits))
        union = list(ref_map.keys() | cur_map.keys())
        if not union:
            continue
        neg_inf = float("-inf")
        ref_vec = [ref_map.get(t, neg_inf) for t in union]
        cur_vec = [cur_map.get(t, neg_inf) for t in union]

        def _softmax(xs):
            mx = max(x for x in xs if x != neg_inf)
            exps = [math.exp(x - mx) if x != neg_inf else 0.0 for x in xs]
            s = sum(exps)
            return [e / s for e in exps] if s > 0 else [0.0] * len(exps)

        p = _softmax(ref_vec)
        q = _softmax(cur_vec)
        # KL(P||Q) = sum P log(P/Q). For p==0 contribute 0.
        kl = 0.0
        for pp, qq in zip(p, q):
            if pp > 0.0 and qq > 0.0:
                kl += pp * math.log(pp / qq)
            elif pp > 0.0 and qq == 0.0:
                # ref mass on a token current never put in top-K: large penalty.
                kl += float("inf")
                break
        if math.isfinite(kl):
            kl_sum += kl
            kl_n += 1
    if kl_n > 0:
        print(
            f"    [quality] approx KL(ref||cur) on top-{top_k} union: "
            f"mean over {kl_n}/{n_compare} steps = {kl_sum / kl_n:.4f} nats"
        )
    else:
        print(
            "    [quality] KL: 所有比较步都出现 ref top-K token 不在 cur top-K，"
            "KL = +inf；说明 sparse vs dense 分布偏移很大。"
        )


if __name__ == "__main__":
    raise SystemExit(main())
