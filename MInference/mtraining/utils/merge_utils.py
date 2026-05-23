# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import copy
import os
import shutil
import subprocess
from typing import List, Optional

import torch
from nnscaler import merge_state_dicts
from transformers import AutoConfig
from transformers.modeling_utils import PreTrainedModel

from .general import fix_model_state_dict

BLOB_DIR = os.environ.get("BLOB_DIR", "/blob")
HF_REPO_DIR = os.path.join(BLOB_DIR, "hf_repos")
STORE_DIR = os.path.join(BLOB_DIR, "mtrain_expr_data_store")
print(f"STORE_DIR: {STORE_DIR}", flush=True)


def model_to_hf_config_files(model_id: str):
    if "phi" in model_id.lower():
        return [
            "added_tokens.json",
            "config.json",
            "configuration_phi3.py",
            "generation_config.json",
            "modeling_phi3.py",
            "generation_config.json",
            "model.safetensors.index.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
        ]
    elif "qwen" in model_id.lower():
        return [
            "config.json",
            "generation_config.json",
            "modeling_qwen2.py",
            "model.safetensors.index.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
        ]
    elif "llama" in model_id.lower():
        return [
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "model.safetensors.index.json",
            "modeling_llama.py",
            "tokenizer_config.json",
            "tokenizer.json",
        ]
    else:
        raise ValueError(f"Model id: {model_id} unsupported yet.")


FLEXIBLE_FIELDS = ["save_plan_path", "gen_savedir"]


def recursive_equiv_state_dict(dict1, dict2):
    if dict1.keys() != dict2.keys():
        # print out which keys are different
        print("Keys differ:", set(dict1.keys()).symmetric_difference(set(dict2.keys())))
        return False

    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            if not recursive_equiv_state_dict(dict1[key], dict2[key]):
                return False
        elif dict1[key] != dict2[key] and not key in FLEXIBLE_FIELDS:
            print(
                f"{__name__} | Values differ at key '{key}': {dict1[key]} != {dict2[key]}"
            )
            return False
    return True


def merge_checkpoint(checkpoint_files: List[str], output_file: str):
    print(f"merge_checkpoint | Start merging ckpt files...", flush=True)
    # state_dicts = [torch.load(f, map_location='cpu') for f in checkpoint_files]
    state_dicts = []
    for i, file_path in enumerate(checkpoint_files):
        print(
            f"merge_checkpoint | Loading {i}-th checkpoint from {file_path}...",
            flush=True,
        )
        state_dict = torch.load(file_path, map_location="cpu")
        state_dicts.append(state_dict)

    train_args_0 = copy.deepcopy(state_dicts[0]["train_args"])
    train_args_0.pop("gen_savedir")
    for i in range(1, len(state_dicts)):
        train_args_i = copy.deepcopy(state_dicts[i]["train_args"])
        train_args_i.pop("gen_savedir")

        # if train_args_i != train_args_0:
        if not recursive_equiv_state_dict(train_args_i, train_args_0):
            raise ValueError(
                f"train_args in {checkpoint_files[i]} is different from {checkpoint_files[0]}"
            )

    print(f"merge_checkpoint | Invoking nnscaler.merge_state_dicts...", flush=True)
    module_state_dict, opt_state_dict = merge_state_dicts(
        [s["model"] for s in state_dicts], [s["optimizer"] for s in state_dicts]
    )
    train_args = copy.deepcopy(state_dicts[0]["train_args"])
    train_args["checkpoint"]["save_type"] = "merged"
    merged_state_dict = {
        "model": module_state_dict,
        "optimizer": opt_state_dict,
        "lr_scheduler": state_dicts[0].get("lr_scheduler", None),
        "train_status": state_dicts[0]["train_status"],
        "train_args": train_args,
        "rng_states": None,
    }

    print(f"merge_checkpoint | Saving merged state_dict into {output_file}", flush=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(merged_state_dict, output_file)


def load_ckpt_files(ckpt_dir):
    print(f"load_ckpt_files | Looking for ckpt files in {ckpt_dir}", flush=True)
    ckpt_files = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".ckpt") and not f.startswith("merged")
    ]

    if "rank_" in ckpt_dir.split("/")[-2]:
        rank = int(ckpt_dir.split("/")[-2].split("_")[-1])

        while os.path.exists(ckpt_dir.replace(f"rank_{rank}", f"rank_{rank+1}")):
            ckpt_dir = ckpt_dir.replace(f"rank_{rank}", f"rank_{rank+1}")
            print(f"load_ckpt_files | Looking for ckpt files in {ckpt_dir}", flush=True)
            ckpt_files += [
                os.path.join(ckpt_dir, f)
                for f in os.listdir(ckpt_dir)
                if f.endswith(".ckpt") and not f.startswith("merged")
            ]
            rank += 1

    return ckpt_files


def merge_ckpts(
    expr_data_dir: str,
    epoch_idx,
    iter_idx,
    override: bool = False,
):
    # --------------------------------------------------------------------------
    # Set ckpt save path and URL
    ckpt_dir = os.path.join(
        expr_data_dir,
        "checkpoints",
        f"{epoch_idx:04d}-{iter_idx:04d}",
    )
    if not os.path.exists(ckpt_dir):
        # Legacy Storage Path
        ckpt_dir = os.path.join(
            expr_data_dir,
            "checkpoints",
            "rank_0",
            f"{epoch_idx:04d}-{iter_idx:04d}",
        )

    # example: /scratch/sync/nnscaler_store/A100_4/minfer_qwen/qwen_moba_zigzag_mini_262144/merged_ckpts/0000-0005/pytorch_model.bin
    merged_ckpt_save_dir = os.path.join(
        expr_data_dir,
        "merged_ckpts",
        f"{epoch_idx:04d}-{iter_idx:04d}",
    )
    os.makedirs(merged_ckpt_save_dir, exist_ok=True)
    merged_ckpt_save_path = os.path.join(merged_ckpt_save_dir, "pytorch_model.bin")
    if os.path.exists(merged_ckpt_save_path) and not override:
        print(
            f"merge_ckpts | Checkpoint already exists in {merged_ckpt_save_path} and skip merging (enabling --override to override)"
        )
        return True

    print(
        f"merge_ckpts | Merging ckpt files in {ckpt_dir} to {merged_ckpt_save_path}",
        flush=True,
    )
    ckpt_files = load_ckpt_files(ckpt_dir)
    merge_checkpoint(ckpt_files, merged_ckpt_save_path)
    print(f"merge_ckpts | Merged ckpt files.", flush=True)

    # --------------------------------------------------------------------------
    print(
        f"merge_ckpts | Converting merged checkpoint to correct format...", flush=True
    )
    ckpt_data = torch.load(merged_ckpt_save_path)
    ckpt_model_data = ckpt_data.pop("model")
    ckpt_model_data = {k[6:]: v for k, v in ckpt_model_data.items()}
    torch.save(ckpt_model_data, merged_ckpt_save_path)
    print(f"merge_ckpts | Checkpoint merged.", flush=True)

    return True


def run_cmd(cmd):
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Error running command {cmd}. Exiting...")
        return 1

    return 0


def transfer_by_cp(local_path, dir="upload", source=None):
    remote_path = local_path.replace("/scratch/eval", STORE_DIR)
    if dir == "upload":
        os.makedirs(os.path.dirname(remote_path), exist_ok=True)
        run_res = run_cmd(["cp", local_path, remote_path])
    else:
        if source == None:
            run_res = run_cmd(["cp", remote_path, local_path])
        else:
            run_res = run_cmd(["cp", source, local_path])

    if run_res != 0:
        print(
            f"Error {'uploading' if dir == 'upload' else 'downloading'} {local_path}. Exiting..."
        )
        return 1
    return 0


def copy_configs_to_merged_dir(
    expr_data_dir: str,
    epoch_idx: int,
    iter_idx: int,
    model_id: str,
    override: bool = False,
):
    merged_ckpt_save_dir = os.path.join(
        expr_data_dir, "merged_ckpts", f"{epoch_idx:04d}-{iter_idx:04d}"
    )
    merged_ckpt_save_path = os.path.join(merged_ckpt_save_dir, "pytorch_model.bin")
    if not os.path.exists(merged_ckpt_save_path):
        raise ValueError(
            f"Merged checkpoint path {merged_ckpt_save_path} does not exist in {expr_data_dir}. Run merge_ckpt first."
        )

    print("-" * 20)
    print(
        f"copy_configs_to_merged_dir | Copying model files to {merged_ckpt_save_dir}",
        flush=True,
    )
    model_name = model_id.split("/")[-1]
    config_files = model_to_hf_config_files(model_id)
    for file in config_files:
        hf_file_path = os.path.join(HF_REPO_DIR, model_name, file)
        merged_path = os.path.join(merged_ckpt_save_dir, file)
        if os.path.exists(merged_path) and not override:
            print(f"\tFile {merged_path} already exists. Use -o to override.")
            continue

        print(f"\tCopying {hf_file_path} to {merged_path}...")
        shutil.copyfile(hf_file_path, merged_path)

    print(f"Config files copied successfully.")
    return True


def load_merged_model(
    model_cls: PreTrainedModel,
    expr_data_dir: str,
    epoch_idx: int,
    iter_idx: int,
    merged_ckpt_dir: Optional[str] = None,
):
    if merged_ckpt_dir is None:
        model_dir = os.path.join(
            expr_data_dir, "merged_ckpts", f"{epoch_idx:04d}-{iter_idx:04d}"
        )
        model_config = AutoConfig.from_pretrained(model_dir)
        if "mini" in model_dir:
            model_config.num_hidden_layers = 2
    else:
        model_dir = merged_ckpt_dir
        model_config = AutoConfig.from_pretrained(model_dir)
        if "mini" in model_dir:
            model_config.num_hidden_layers = 2

    model = model_cls(config=model_config)
    model_state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"))
    if len(list(model_state_dict.keys())[0].split(".")) == 1:
        # For Ring-Attention models, the merged checkpoint is directly copied from one of the shards and has different key names.
        model_state_dict = fix_model_state_dict(model, model_state_dict)

    model.load_state_dict(model_state_dict)
    return model
