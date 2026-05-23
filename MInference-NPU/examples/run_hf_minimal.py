# Copyright (c) 2026
# Licensed under The MIT License [see LICENSE for details]
"""最小 HF 调用示例 —— 用 MInference-NPU 在 Ascend 910B 上跑 Llama / Qwen 长上下文 prefill + 解码 ~10 token。

用法：
    # 默认跑 Qwen2.5-7B-Instruct（128k 配置），单卡 npu:0
    python examples/run_hf_minimal.py

    # 指定模型 / 长度 / 多卡
    python examples/run_hf_minimal.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --ctx-len 32768 \
        --device-map auto

注意：
- 跑大模型需要 HF 远端权重（首次会下载十几 GB）。建议提前 `huggingface-cli download` 缓存。
- v1 阶段三种稀疏分支都退化为 dense，因此 latency 与 HF eager attention 相当 —— 不要拿这
  个数字评估 MInference 加速效果。M2-M4 完成后再做对比。
- 长上下文测试如果 OOM，先把 `--ctx-len` 调小到 8192 试通链路，再逐步放大。
"""

from __future__ import annotations

import argparse
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="MInference-NPU 最小 HF 示例")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF 模型名（要在 MODEL2PATH 里有对应的 best_pattern JSON）",
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
        help="解码 token 数（v1 dense fallback 阶段不必跑太长）",
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
    args = parser.parse_args()

    import torch

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("[ERROR] torch_npu 未安装，本示例必须在昇腾 NPU 机器上跑")
        return 1

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from minference import MInference

    print(f"[1/4] 加载 tokenizer & model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
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
            model_name=args.model if args.attn_type == "minference" else None,
        )(model)
    else:
        print("[2/4] 跳过 patch (attn_type='hf')")

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
    print(f"    实际 prompt 长度：{input_ids.shape[1]} tokens")

    print(f"[4/4] generate(max_new_tokens={args.max_new_tokens})")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids.to(next(model.parameters()).device),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    dt = time.time() - t0
    new_tokens = out[0, input_ids.shape[1]:]
    print(f"    完成，用时 {dt:.2f}s，解码 {len(new_tokens)} tokens")
    print(f"    输出文本：{tokenizer.decode(new_tokens, skip_special_tokens=True)!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
