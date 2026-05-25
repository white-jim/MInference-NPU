# M5 第二轮测试结果

## 2026-05-25 HF 端到端 8K minference

### 命令

```bash
python examples/run_hf_minimal.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-path /data/guoshiyao/resources/models/Meta-Llama-3.1-8B-Instruct \
  --ctx-len 8192 \
  --max-new-tokens 16 \
  --device-map npu:0 \
  --attn-type minference 2>&1 | tee /tmp/run_minf_8k.log
```

### 显存占用

```text
17902MB - 18354MB
```

### 原始日志

```text
/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/8.1.RC1 owner does not match the current owner.
  warnings.warn(f"Warning: The {path} owner does not match the current owner.")
/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/8.1.RC1/aarch64-linux/ascend_toolkit_install.info owner does not match the current owner.
  warnings.warn(f"Warning: The {path} owner does not match the current owner.")
`torch_dtype` is deprecated! Use `dtype` instead!
[1/4] 加载 tokenizer & model: /data/guoshiyao/resources/models/Meta-Llama-3.1-8B-Instruct  (best_pattern key=meta-llama/Llama-3.1-8B-Instruct)
Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
Loading checkpoint shards:  25%|██▌       | 1/4 [00:00<00:02,  1.10it/s]
Loading checkpoint shards:  50%|█████     | 2/4 [00:01<00:01,  1.21it/s]
Loading checkpoint shards:  75%|███████▌  | 3/4 [00:02<00:00,  1.28it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:02<00:00,  1.81it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:02<00:00,  1.53it/s]
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
[2/4] 应用 MInference patch (attn_type=minference)
<---- MInference Config Detail ----> attn_type minference, kv_type dense
[MInference-NPU] Patched LlamaAttention.forward (starting_layer=-1, dense fallback for all sparse modes)
[3/4] 构造长 prompt（ctx_len=8192）
    实际 prompt 长度：8192 tokens
[4/4] generate(max_new_tokens=16)
    完成，用时 308.53s，解码 16 tokens
    输出文本：' fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog'
```

## 2026-05-25 HF 端到端 16K minference

### 命令

```bash
python examples/run_hf_minimal.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-path /data/guoshiyao/resources/models/Meta-Llama-3.1-8B-Instruct \
  --ctx-len 16384 \
  --max-new-tokens 16 \
  --device-map npu:0 \
  --attn-type minference 2>&1 | tee /tmp/run_minf_16k.log
```

### 显存占用

```text
20247MB
```

### 原始日志

```text
/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/8.1.RC1 owner does not match the current owner.
  warnings.warn(f"Warning: The {path} owner does not match the current owner.")
/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/8.1.RC1/aarch64-linux/ascend_toolkit_install.info owner does not match the current owner.
  warnings.warn(f"Warning: The {path} owner does not match the current owner.")
`torch_dtype` is deprecated! Use `dtype` instead!
[1/4] 加载 tokenizer & model: /data/guoshiyao/resources/models/Meta-Llama-3.1-8B-Instruct  (best_pattern key=meta-llama/Llama-3.1-8B-Instruct)
Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
Loading checkpoint shards:  25%|██▌       | 1/4 [00:00<00:02,  1.09it/s]
Loading checkpoint shards:  50%|█████     | 2/4 [00:01<00:01,  1.20it/s]
Loading checkpoint shards:  75%|███████▌  | 3/4 [00:02<00:00,  1.28it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:02<00:00,  1.81it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:02<00:00,  1.53it/s]
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
[2/4] 应用 MInference patch (attn_type=minference)
<---- MInference Config Detail ----> attn_type minference, kv_type dense
[MInference-NPU] Patched LlamaAttention.forward (starting_layer=-1, dense fallback for all sparse modes)
[3/4] 构造长 prompt（ctx_len=16384）
    实际 prompt 长度：16384 tokens
[4/4] generate(max_new_tokens=16)
    完成，用时 616.32s，解码 16 tokens
    输出文本：' lazy dog. The quick brown fox jumps over the lazy dog. The quick brown'
```

## 2026-05-25 HF 端到端 16K dense

### 命令

```bash
python examples/run_hf_minimal.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-path /data/guoshiyao/resources/models/Meta-Llama-3.1-8B-Instruct \
  --ctx-len 16384 \
  --max-new-tokens 16 \
  --device-map npu:0 \
  --attn-type dense 2>&1 | tee /tmp/run_dense_16k.log
```

### 显存占用

```text
14535MB - 20951MB
```

### 原始日志

```text
/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/8.1.RC1 owner does not match the current owner.
  warnings.warn(f"Warning: The {path} owner does not match the current owner.")
/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/8.1.RC1/aarch64-linux/ascend_toolkit_install.info owner does not match the current owner.
  warnings.warn(f"Warning: The {path} owner does not match the current owner.")
`torch_dtype` is deprecated! Use `dtype` instead!
[1/4] 加载 tokenizer & model: /data/guoshiyao/resources/models/Meta-Llama-3.1-8B-Instruct  (best_pattern key=meta-llama/Llama-3.1-8B-Instruct)
Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
Loading checkpoint shards:  25%|██▌       | 1/4 [00:00<00:02,  1.08it/s]
Loading checkpoint shards:  50%|█████     | 2/4 [00:01<00:01,  1.21it/s]
Loading checkpoint shards:  75%|███████▌  | 3/4 [00:02<00:00,  1.28it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:02<00:00,  1.82it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:02<00:00,  1.54it/s]
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
[2/4] 应用 MInference patch (attn_type=dense)
<---- MInference Config Detail ----> attn_type dense, kv_type dense
[MInference-NPU] Patched LlamaAttention.forward (starting_layer=-1, dense fallback for all sparse modes)
[3/4] 构造长 prompt（ctx_len=16384）
    实际 prompt 长度：16384 tokens
[4/4] generate(max_new_tokens=16)
    完成，用时 4.57s，解码 16 tokens
    输出文本：' lazy dog. The quick brown fox jumps over the lazy dog. The quick brown'
```
