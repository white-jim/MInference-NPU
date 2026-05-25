# M5 测试结果

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
17675MB
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
Loading checkpoint shards:  25%|██▌       | 1/4 [00:01<00:03,  1.00s/it]
Loading checkpoint shards:  50%|█████     | 2/4 [00:01<00:01,  1.10it/s]
Loading checkpoint shards:  75%|███████▌  | 3/4 [00:02<00:00,  1.13it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:02<00:00,  1.63it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:02<00:00,  1.38it/s]
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
    完成，用时 309.74s，解码 16 tokens
    输出文本：' fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog'
```

## 2026-05-25 HF 端到端 32K attn_type 对比

### 命令

```bash
for AT in minference dense hf; do
  echo "=========== attn_type=$AT ===========" | tee -a /tmp/run_cmp_32k.log
  python examples/run_hf_minimal.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --model-path /data/guoshiyao/resources/models/Meta-Llama-3.1-8B-Instruct \
    --ctx-len 32768 \
    --max-new-tokens 16 \
    --device-map npu:0 \
    --attn-type $AT 2>&1 | tee -a /tmp/run_cmp_32k.log
done
```

### 原始日志

```text
=========== attn_type=minference ===========
/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/8.1.RC1 owner does not match the current owner.
  warnings.warn(f"Warning: The {path} owner does not match the current owner.")
/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/8.1.RC1/aarch64-linux/ascend_toolkit_install.info owner does not match the current owner.
  warnings.warn(f"Warning: The {path} owner does not match the current owner.")
`torch_dtype` is deprecated! Use `dtype` instead!
[1/4] 加载 tokenizer & model: /data/guoshiyao/resources/models/Meta-Llama-3.1-8B-Instruct  (best_pattern key=meta-llama/Llama-3.1-8B-Instruct)
Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
Loading checkpoint shards:  25%|██▌       | 1/4 [00:00<00:02,  1.09it/s]
Loading checkpoint shards:  50%|█████     | 2/4 [00:01<00:01,  1.22it/s]
Loading checkpoint shards:  75%|███████▌  | 3/4 [00:02<00:00,  1.28it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:02<00:00,  1.81it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:02<00:00,  1.54it/s]
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
/data/guoshiyao/zhw/MInference-NPU/MInference-NPU/minference/modules/minference_forward.py:246: UserWarning: vertical_slash_sparse_attention: 序列长度 32768 > 16384。 bool mask 构建需 O(S²) 内存，退化为 causal dense。 超长序列请改用 Triton-Ascend 路径（路径 B）。
  return _vs_sparse_attention(q, k, v, v_idx, s_idx)
[2/4] 应用 MInference patch (attn_type=minference)
<---- MInference Config Detail ----> attn_type minference, kv_type dense
[MInference-NPU] Patched LlamaAttention.forward (starting_layer=-1, dense fallback for all sparse modes)
[3/4] 构造长 prompt（ctx_len=32768）
    实际 prompt 长度：32768 tokens
[4/4] generate(max_new_tokens=16)
    完成，用时 12.17s，解码 16 tokens
    输出文本：' over the lazy dog. The quick brown fox jumps over the lazy dog. The'
=========== attn_type=dense ===========
/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/8.1.RC1 owner does not match the current owner.
  warnings.warn(f"Warning: The {path} owner does not match the current owner.")
/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/8.1.RC1/aarch64-linux/ascend_toolkit_install.info owner does not match the current owner.
  warnings.warn(f"Warning: The {path} owner does not match the current owner.")
`torch_dtype` is deprecated! Use `dtype` instead!
[1/4] 加载 tokenizer & model: /data/guoshiyao/resources/models/Meta-Llama-3.1-8B-Instruct  (best_pattern key=meta-llama/Llama-3.1-8B-Instruct)
Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
Loading checkpoint shards:  25%|██▌       | 1/4 [00:00<00:02,  1.11it/s]
Loading checkpoint shards:  50%|█████     | 2/4 [00:01<00:01,  1.23it/s]
Loading checkpoint shards:  75%|███████▌  | 3/4 [00:02<00:00,  1.29it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:02<00:00,  1.82it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:02<00:00,  1.55it/s]
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
[2/4] 应用 MInference patch (attn_type=dense)
<---- MInference Config Detail ----> attn_type dense, kv_type dense
[MInference-NPU] Patched LlamaAttention.forward (starting_layer=-1, dense fallback for all sparse modes)
[3/4] 构造长 prompt（ctx_len=32768）
    实际 prompt 长度：32768 tokens
[4/4] generate(max_new_tokens=16)
    完成，用时 10.54s，解码 16 tokens
    输出文本：' over the lazy dog. The quick brown fox jumps over the lazy dog. The'
=========== attn_type=hf ===========
/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/8.1.RC1 owner does not match the current owner.
  warnings.warn(f"Warning: The {path} owner does not match the current owner.")
/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/ascend-toolkit/8.1.RC1/aarch64-linux/ascend_toolkit_install.info owner does not match the current owner.
  warnings.warn(f"Warning: The {path} owner does not match the current owner.")
`torch_dtype` is deprecated! Use `dtype` instead!
[1/4] 加载 tokenizer & model: /data/guoshiyao/resources/models/Meta-Llama-3.1-8B-Instruct  (best_pattern key=meta-llama/Llama-3.1-8B-Instruct)
Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
Loading checkpoint shards:  25%|██▌       | 1/4 [00:00<00:02,  1.08it/s]
Loading checkpoint shards:  50%|█████     | 2/4 [00:01<00:01,  1.22it/s]
Loading checkpoint shards:  75%|███████▌  | 3/4 [00:02<00:00,  1.27it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:02<00:00,  1.81it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:02<00:00,  1.53it/s]
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
[2/4] 跳过 patch (attn_type='hf')
[3/4] 构造长 prompt（ctx_len=32768）
    实际 prompt 长度：32768 tokens
[4/4] generate(max_new_tokens=16)
Traceback (most recent call last):
  File "/data/guoshiyao/zhw/MInference-NPU/MInference-NPU/examples/run_hf_minimal.py", line 133, in <module>
    raise SystemExit(main())
  File "/data/guoshiyao/zhw/MInference-NPU/MInference-NPU/examples/run_hf_minimal.py", line 119, in main
    out = model.generate(
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/generation/utils.py", line 2564, in generate
    result = decoding_method(
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/generation/utils.py", line 2784, in _sample
    outputs = self(**model_inputs, return_dict=True)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/utils/generic.py", line 918, in wrapper
    output = func(self, *args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 459, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/utils/generic.py", line 1072, in wrapper
    outputs = func(self, *args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 395, in forward
    hidden_states = decoder_layer(
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/modeling_layers.py", line 94, in __call__
    return super().__call__(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 294, in forward
    hidden_states, _ = self.self_attn(
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 252, in forward
    attn_output, attn_weights = attention_interface(
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 184, in eager_attention_forward
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
RuntimeError: NPU out of memory. Tried to allocate 64.00 GiB (NPU 0; 60.96 GiB total capacity; 18.35 GiB already allocated; 18.35 GiB current active; 41.44 GiB free; 19.17 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.
[ERROR] 2026-05-25-15:55:55 (PID:2420485, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
```

## 2026-05-25 HF 端到端 128K 多卡 minference

### 命令

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  python examples/run_hf_minimal.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --model-path /data/guoshiyao/resources/models/Meta-Llama-3.1-8B-Instruct \
    --ctx-len 131072 \
    --max-new-tokens 16 \
    --device-map auto \
    --attn-type minference 2>&1 | tee /tmp/run_minf_128k.log
```

### 显存占用

```text
| NPU | Chip | PID     | Process | Memory(MB) |
| 0   | 0    | 2426891 | python  | 58185      |
| 1   | 0    | 2426891 | python  | 4277       |
| 2   | 0    | 2426891 | python  | 4277       |
| 3   | 0    | 2426891 | python  | 5357       |
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
Loading checkpoint shards:  25%|██▌       | 1/4 [00:00<00:02,  1.07it/s]
Loading checkpoint shards:  50%|█████     | 2/4 [00:01<00:01,  1.01it/s]
Loading checkpoint shards:  75%|███████▌  | 3/4 [00:02<00:00,  1.00it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:03<00:00,  1.50it/s]
Loading checkpoint shards: 100%|██████████| 4/4 [00:03<00:00,  1.28it/s]
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
/data/guoshiyao/zhw/MInference-NPU/MInference-NPU/minference/modules/minference_forward.py:246: UserWarning: vertical_slash_sparse_attention: 序列长度 131072 > 16384。 bool mask 构建需 O(S²) 内存，退化为 causal dense。 超长序列请改用 Triton-Ascend 路径（路径 B）。
  return _vs_sparse_attention(q, k, v, v_idx, s_idx)
[2/4] 应用 MInference patch (attn_type=minference)
<---- MInference Config Detail ----> attn_type minference, kv_type dense
[MInference-NPU] Patched LlamaAttention.forward (starting_layer=-1, dense fallback for all sparse modes)
[3/4] 构造长 prompt（ctx_len=131072）
    实际 prompt 长度：131072 tokens
[4/4] generate(max_new_tokens=16)
Traceback (most recent call last):
  File "/data/guoshiyao/zhw/MInference-NPU/MInference-NPU/examples/run_hf_minimal.py", line 133, in <module>
    raise SystemExit(main())
  File "/data/guoshiyao/zhw/MInference-NPU/MInference-NPU/examples/run_hf_minimal.py", line 119, in main
    out = model.generate(
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/generation/utils.py", line 2564, in generate
    result = decoding_method(
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/generation/utils.py", line 2784, in _sample
    outputs = self(**model_inputs, return_dict=True)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/accelerate/hooks.py", line 170, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/utils/generic.py", line 918, in wrapper
    output = func(self, *args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 459, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/utils/generic.py", line 1072, in wrapper
    outputs = func(self, *args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 395, in forward
    hidden_states = decoder_layer(
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/modeling_layers.py", line 94, in __call__
    return super().__call__(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/accelerate/hooks.py", line 170, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py", line 294, in forward
    hidden_states, _ = self.self_attn(
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/guoshiyao/miniforge3/envs/flexhead/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/guoshiyao/zhw/MInference-NPU/MInference-NPU/minference/modules/minference_forward.py", line 430, in forward
    attn_output = gather_last_q_vertical_slash_topk_v4(
  File "/data/guoshiyao/zhw/MInference-NPU/MInference-NPU/minference/modules/minference_forward.py", line 276, in gather_last_q_vertical_slash_topk_v4
    return _vertical_and_slash_kernel(self, q, k, v, vertical_size, slash_size)
  File "/data/guoshiyao/zhw/MInference-NPU/MInference-NPU/minference/modules/minference_forward.py", line 246, in _vertical_and_slash_kernel
    return _vs_sparse_attention(q, k, v, v_idx, s_idx)
  File "/data/guoshiyao/zhw/MInference-NPU/MInference-NPU/minference/ops/vertical_slash_kernel_npu.py", line 343, in vertical_slash_sparse_attention
    out = dense_attention(query, key, value, causal=True)
  File "/data/guoshiyao/zhw/MInference-NPU/MInference-NPU/minference/backend_npu/attention.py", line 131, in dense_attention
    causal_mask = torch.ones(s_q, s_k, device=q.device, dtype=torch.bool).triu(
RuntimeError: NPU out of memory. Tried to allocate 16.00 GiB (NPU 0; 60.96 GiB total capacity; 41.61 GiB already allocated; 41.61 GiB current active; 8.56 GiB free; 52.04 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.
[ERROR] 2026-05-25-15:57:35 (PID:2426891, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
```
