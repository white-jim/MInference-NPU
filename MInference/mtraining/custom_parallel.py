# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import inspect
import os
import shutil
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
import torch.distributed as dist
from nnscaler.autodist.apis import parallelize_graph
from nnscaler.autodist.autodist_config import AutoDistConfig
from nnscaler.autodist.util import get_default_profile_path
from nnscaler.graph import IRGraph
from nnscaler.graph.parser import FxModuleParser
from nnscaler.parallel import (
    _FORWARD_ARGS_DUMP_FILE,
    _GENCODE_FILE_TEMPLATE,
    _GRAPH_DUMP_FILE,
    _PREDEFINED_POLICIES,
    BroadcastGenFilesStrategy,
    ComputeConfig,
    RegenStatus,
    ReuseType,
    _broadcast_gen_files,
    _clean_files,
    _compile_flags,
    _gencode,
    _is_any_gencode_loaded,
    _load_parallel_module_class,
    _prepare_namespace,
)
from nnscaler.runtime.device import DeviceGroup
from nnscaler.runtime.module import (
    AttrMeta,
    CubeModule,
    ExtraState,
    OriginModuleMetadata,
    ParallelModule,
)

_CUSTOM_PREDEFINED_POLICIES: Dict[
    str, Callable[[IRGraph, "ComputeConfig"], IRGraph]
] = {}
for k, v in _PREDEFINED_POLICIES.items():
    if k != "autodist":
        _CUSTOM_PREDEFINED_POLICIES[k] = v

import logging

logger = logging.getLogger(__name__)


def pas_autodist(graph: IRGraph, cfg: "ComputeConfig") -> IRGraph:
    print(f"{__name__} | Using custom autodist policy defined in {__file__}")
    pas_cfg = cfg.pas_config

    update_freq = pas_cfg.get("update_freq", 1)
    if isinstance(update_freq, (tuple, list)):
        update_freq = update_freq[0]

    # optional parameters
    explore_pipeline = pas_cfg.get("explore_pipeline", False)
    if explore_pipeline and not cfg.use_end2end:
        raise ValueError("explore_pipeline cannot be enabled if use_end2end is False")
    if explore_pipeline and cfg.use_async_reducer:
        raise ValueError(
            "explore_pipeline cannot be enabled if use_async_reducer is True"
        )

    pipeline_scheduler = pas_cfg.get("pipeline_scheduler", "1f1b")
    if pipeline_scheduler != "1f1b":
        raise ValueError(f"Only 1f1b scheduler is supported in autodist.")

    mesh_col = pas_cfg.get("max_partition_degree", cfg.plan_ngpus)
    if cfg.plan_ngpus % mesh_col != 0:
        raise ValueError(
            f"plan_ngpus {cfg.plan_ngpus} should be divisible by max_partition_degree {mesh_col}"
        )
    mesh_row = cfg.plan_ngpus // mesh_col
    if not explore_pipeline and mesh_row != 1:
        raise ValueError("mesh_row should be 1 if pipeline is not enabled")
    memory_constraint = pas_cfg.get("mem_constraint", -1)
    task_name = pas_cfg.get("task_name", "_")
    use_memory_efficient_fp16 = pas_cfg.get("use_memory_efficient_fp16", False)
    use_memory_efficient_bf16 = pas_cfg.get("use_memory_efficient_bf16", False)
    use_fp16 = pas_cfg.get("use_fp16", use_memory_efficient_fp16)
    use_bf16 = pas_cfg.get("use_bf16", use_memory_efficient_bf16)
    re_profile = pas_cfg.get("re_profile", False)
    verbose = pas_cfg.get("verbose", False)
    load_plan_path = pas_cfg.get("load_plan_path", None)
    save_plan_path = pas_cfg.get("save_plan_path", None)
    partition_constraints_path = pas_cfg.get("partition_constraints_path", "")
    recompute_modules = pas_cfg.get("recompute_modules", "")
    pipeline_pivots = pas_cfg.get("pipeline_pivots", "")
    use_apex_fused_adam_v2 = pas_cfg.get("use_apex_fused_adam_v2", False)
    parallel_profile = pas_cfg.get("parallel_profile", True)
    transient_mem_coef = pas_cfg.get("transient_mem_coef", 2)
    profile_dir = pas_cfg.get("profile_dir", get_default_profile_path())
    solver = pas_cfg.get("solver", "dp")

    task_name = f"{task_name}_{cfg.plan_ngpus}gpus_{update_freq}update_freq"
    if memory_constraint == -1:
        # consider memory fragmentation and other buffers, use 80% of the memory
        memory_constraint = int(0.8 * torch.cuda.mem_get_info()[1] / 1024 / 1024 / 1024)
    if cfg.use_zero:
        zero_stage = 1
        zero_ngroups = cfg.zero_ngroups
    else:
        zero_stage = 0
        zero_ngroups = 1
    if use_fp16 or use_bf16:
        support_inkernel_cast = use_apex_fused_adam_v2
        if use_memory_efficient_fp16 or use_memory_efficient_bf16:
            # Check fairseq/optim/fused_adam.py
            # If memory efficient:
            # Considered in opt_resident_mem: fp32 moment1, fp32 moment2.
            # Considered in opt_transient_mem: fp32 weight, fp32 gradient,
            # because fp16 weight and gradient are casted to fp32.
            # Here weight_mem is in fp16, so multiply by (2+2).
            opt_resident_coef = 4
            opt_transient_coef = 0 if support_inkernel_cast else 4
        else:
            # If not memory efficient:
            # Considered in opt_resident_mem: fp32 moment1, fp32 moment2, fp32 weight.
            # Considered in opt_transient_mem: fp32 gradient,
            # because fp16 gradient are casted to fp32.
            # Here weight_mem is in fp16, so multiply by (2+2+2).
            opt_resident_coef = 6
            # inkernel cast between fp32 weight and fp16 grad has not support
            opt_transient_coef = 2 if support_inkernel_cast else 2
    else:
        # Considered in opt_resident_mem: fp32 moment1, fp32 moment2
        # Considered in opt_transient_mem: 0
        # Here weight_mem is in fp32, so multiply by (1+1).
        opt_resident_coef = 2
        opt_transient_coef = 0

    autodist_cfg = AutoDistConfig(
        mesh_row=mesh_row,
        mesh_col=mesh_col,
        update_freq=update_freq,
        task_name=task_name,
        profile_dir=profile_dir,
        is_train=not cfg.inference_only,
        ignore_small_tensor_threshold=524288,  # 0.5 MB is a good threshold to reduce search time and make the result correct, will refine later
        memory_granularity=524288,  # 0.5 MB is a good threshold to reduce search time and make the result correct, will refine later
        consider_mem=True,
        partition_constraints_path=partition_constraints_path,
        memory_constraint=memory_constraint,
        opt_resident_coef=opt_resident_coef,
        opt_transient_coef=opt_transient_coef,
        verbose=verbose,
        re_profile=re_profile,
        world_size=cfg.runtime_ngpus,
        recompute_modules=recompute_modules,
        zero_stage=zero_stage,
        zero_ngroups=zero_ngroups,
        load_plan_path=load_plan_path,
        save_plan_path=save_plan_path,
        pipeline=explore_pipeline,
        pipeline_pivots=pipeline_pivots,
        parallel_profile=parallel_profile,
        transient_mem_coef=transient_mem_coef,
        solver=solver,
    )
    logger.info(f"{__name__} | Using autodist config: {autodist_cfg}")

    return parallelize_graph(graph, autodist_cfg)


_CUSTOM_PREDEFINED_POLICIES["autodist"] = pas_autodist


def compute_config_safe_equals(
    a: Optional["ComputeConfig"], b: Optional["ComputeConfig"]
) -> bool:
    """
    Return False if a and b are from incompatible version of ComputeConfig
    This is only for backward compatibility, and will be removed in future
    and can use `==` when we save dict version of ComputeConfig to file.
    """
    res = True
    try:
        for key in a.__dataclass_fields__:
            if getattr(a, key) != getattr(b, key):
                print(
                    f"{key} not equal: {getattr(a, key)} (old_config) != {getattr(b, key)} (current_config)"
                )

                if key == "user_config":
                    continue
                else:
                    print(
                        f"compute_config_safe_equals | {key} not equal: {getattr(a, key)} (old_config) != {getattr(b, key)} (current_config)"
                    )
                    res = False
        return res
    except AttributeError:
        logger.warning(
            f"compute_config_safe_equals | Failed to compare ComputeConfig. They are incompatible."
            f"Old config: {a}\n"
            f"New config: {b}\n"
        )
        return False


GRAPH_CONFIG_FIELDS = [
    "constant_folding",
    "user_config",
    "inference_only",
    "end2end_mode",
    "trace_strategy",
]


def graph_config_equals(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """
    Return False if a and b are from incompatible version of ComputeConfig
    This is only for backward compatibility, and will be removed in future
    and can use `==` when we save dict version of ComputeConfig to file.
    """
    res = True
    try:
        for key in GRAPH_CONFIG_FIELDS:
            if a[key] != b[key]:
                print(
                    f"graph_config_equals | {key} not equal: {getattr(a, key)} (old_config) != {getattr(b, key)} (current_config)"
                )
                if key != "user_config":
                    res = False
        return res
    except KeyError as e:
        import traceback

        logger.warning(
            "graph_config_equals | Failed to compare GraphConfig with exception.\n"
            f"Exception: {traceback.format_exc()}\n"
            f"Old config: {a}\n"
            f"New config: {b}\n"
        )
        return False


TRACE_FILE_EXTENSIONS = [
    FxModuleParser.ATTR_CONTENT_FILE_0,  # init weights file(fullmodel.pt.*),
    FxModuleParser.ATTR_MAP_FILE,  # param name mapping (dist_param_map.pt)\
    _GRAPH_DUMP_FILE,  # graph dump (graph.ckp),
    _FORWARD_ARGS_DUMP_FILE,  # forward args dump(forward_args.pkl),
    ParallelModule.ORIGIN_MODULE_METADATA_FILE,  # origin module metadata (origin_module_metadata.pt),
]


def transfer_metadata(out_dir, transfer_config: Dict[str, Any]):
    transfer_config_dir, transfer_force = (
        transfer_config["transfer_config_dir"],
        transfer_config["transfer_force"],
    )
    if not os.path.exists(transfer_config_dir):
        # if transfer_config_dir is not set, use the default directory
        transfer_config_dir = transfer_config_dir.replace(
            "compile_config/", "compile_config/rank_0/"
        )
    assert os.path.exists(
        transfer_config_dir
    ), f"Source directory {transfer_config_dir} for transferring does not exist"

    for file in os.listdir(transfer_config_dir):
        if file in TRACE_FILE_EXTENSIONS or file.startswith(
            FxModuleParser.ATTR_CONTENT_FILE_STEM
        ):
            src_file = os.path.join(transfer_config_dir, file)
            dst_file = os.path.join(out_dir, file)

            print(
                f"{__name__} | Copying {src_file} to {dst_file} (local_rank={os.getenv('LOCAL_RANK')})"
            )
            if not os.path.exists(dst_file) or transfer_force:
                shutil.copyfile(src_file, dst_file)

            if not os.path.exists(dst_file):
                raise FileNotFoundError(
                    f"{__name__} | Copy failed ({dst_file} does not exist after copying)"
                )

    # Create a file 'transferred.sign' to indicate that the transfer is done
    with open(os.path.join(out_dir, "transferred.sign"), "w") as f:
        f.write("Transferred from " + transfer_config_dir)


def _prepare_and_check_reusable(
    gen_savedir: str,
    module_or_module_class: Union[Type[torch.nn.Module], torch.nn.Module],
    compute_config: ComputeConfig,
    instance_name: Optional[str] = None,
    reuse: ReuseType = ReuseType.MATCH,
    transfer_config: Dict[str, Any] = None,
) -> Tuple[str, bool, bool]:
    """
    Prepare the output directory for code generation, and also check if the existing code is reusable.

    Args:
        gen_savedir (str): the directory to save generated code
        module_or_module_class (Union[Type[torch.nn.Module], torch.nn.Module]): the original module or module class
        compute_config (ComputeConfig): the environment resource
        instance_name (Optional[str]): the instance name of the generated module. If it is None, will use the default name.
        reuse (ReuseType): specify which part can be reused.

    Returns:
        Tuple[str, bool]: the output directory and whether the existing code is reusable.

    Raises:
        RuntimeError: if the existing code is not reusable,
            will raise RuntimeError if the code is not reusable but the module is already loaded.
    """
    namespace, outdir = _prepare_namespace(
        gen_savedir, module_or_module_class, instance_name
    )
    reusable = False
    transferred = False

    config_file = outdir / ParallelModule.COMPUTE_CONFIG_FILE

    # Empty + Transfer -> config match, graph match, tracing file present -> generate code by MATCH or MOO
    # Empty w.o. Transfer -> Empty -> generate code by MATCH or MOO
    has_transferred = os.path.exists(os.path.join(outdir, "transferred.sign"))
    if (
        transfer_config is not None
        and transfer_config.get("transfer_config_dir", None) is not None
        and (not has_transferred or transfer_config["transfer_force"])
    ):
        # transfer_config_dir: Optional[str] = None,
        transfer_metadata(outdir, transfer_config)
        ComputeConfig.safe_dump_to_file(compute_config, config_file)
        transferred = True

    # decision matrix for code generation
    # reuse flag | dir condition(imported, empty, match, unmatched) | action
    # ---------------------------------------------------------
    #   OVERRIDE   | empty           | generate
    #   OVERRIDE   | imported        | raise error
    #   OVERRIDE   | whatever match  | generate
    #   OVERRIDE   | unmatch         | generate
    #   GRAPH      | empty           | generate
    #   GRAPH      | imported        | raise error
    #   GRAPH      | graph match     | reuse graph, and regenerate code
    #   GRAPH      | all match       | reuse graph, and regenerate code
    #   GRAPH      | unmatch         | generate
    #   MATCH      | empty           | generate
    #   MATCH      | match           | reuse(do nothing)
    #   MATCH*     | whatever unmatch| raise error (except when there's no python source code, see below)
    #   MATCH      | imported        | doesn't matter
    #   MOO        | empty           | generate
    #   MOO        | match           | reuse(do nothing)
    #   MOO        | match graph     | reuse graph, and regenerate code
    #   MOO        | imported        | raise error if whatever unmatch
    #  *: The precondition for `except` part is the compute config should match.
    #     you can take it as a continous operation after a failed generation.
    old_config: Optional[ComputeConfig] = ComputeConfig.safe_load_from_file(config_file)
    is_config_match = compute_config_safe_equals(old_config, compute_config)
    is_graph_config_match = old_config is not None and graph_config_equals(
        old_config.graph_config, compute_config.graph_config
    )
    trace_meta_files = [
        outdir
        / FxModuleParser.ATTR_CONTENT_FILE_0,  # init weights file(fullmodel.pt.*),
        outdir / FxModuleParser.ATTR_MAP_FILE,  # param name mapping (dist_param_map.pt)
    ]

    if reuse == ReuseType.MATCH or reuse == ReuseType.MOO:
        # check if the module is already generated
        expected_output_files = [
            outdir / _GENCODE_FILE_TEMPLATE.format(rank)
            for rank in range(compute_config.runtime_ngpus)
        ]
        expected_output_files.extend(trace_meta_files)
        expected_output_files.append(config_file)
        expected_output_files.append(
            outdir / _GRAPH_DUMP_FILE
        )  # graph dump (graph.ckp),
        expected_output_files.append(
            outdir / _FORWARD_ARGS_DUMP_FILE
        )  # forward args dump(forward_args.pkl),
        expected_output_files.append(
            outdir / ParallelModule.ORIGIN_MODULE_METADATA_FILE
        )  # origin module metadata (origin_module_metadata.pt),
        existing_output_files = [
            f
            for f in outdir.glob("*")
            if f.is_file()
            and (  # just take fullmodel.pt.0 to compare
                not f.name.startswith(FxModuleParser.ATTR_CONTENT_FILE_STEM)
                or f.name == FxModuleParser.ATTR_CONTENT_FILE_0
            )
            and not f.name.endswith(".sign")
        ]

        print(f"{__name__} | compute config match: {is_config_match}")
        print(f"{__name__} | graph config match: {is_graph_config_match}")
        print(f"{__name__} | existing output files: {existing_output_files}")
        print(f"{__name__} | expected output files: {expected_output_files}")

        if existing_output_files:  # if the directory is not empty
            if (
                is_config_match
                and all([output_file.exists() for output_file in expected_output_files])
                and len(existing_output_files) == len(expected_output_files)
            ):
                print(f"{__name__} | Reuse existing files in {outdir}")
                reusable = True  # everything is matched.
            elif is_config_match and all(
                f.suffix != ".py" for f in existing_output_files
            ):
                # No python source code is generated.
                # which means its last generation failed.
                # in this case, we can reuse the same directory safely.
                logger.info(
                    f"Output directory {outdir} is not empty. "
                    f"But no python source code is present. "
                    f"Will reuse the directory and the graph dump if present."
                )
                # we have to trace the graph again if not all meta files are present.
                print(
                    f"{__name__} | compute config match but no python code exists in {outdir}"
                )
                if not all([meta_file.exists() for meta_file in trace_meta_files]):
                    print(
                        f"{__name__} | compute config match but no python code exists in {outdir} and not all meta files are present"
                    )
                    _clean_files(outdir)
            elif reuse == ReuseType.MATCH:
                raise RuntimeError(
                    f"Output directory {outdir} is not empty. "
                    f"And the existing files do not match with current config. "
                    f"You can remove the directory and try again, "
                    f"or set reuse to ReuseType.NONE/ReuseType.OVERRIDE to regenerate the code."
                )
            else:
                assert reuse == ReuseType.MOO
                if _is_any_gencode_loaded(namespace):
                    raise RuntimeError(
                        f"Output directory {outdir} is already loaded. "
                        f"You can not override a loaded module."
                    )
                elif is_graph_config_match:
                    # reuse the graph dump
                    print(
                        f"{__name__} | MOO | graph match -> reuse graph but clean the current code"
                    )
                    _clean_files(outdir, "*.py")
                else:
                    _clean_files(outdir)
    else:
        # check if the module is already loaded
        if _is_any_gencode_loaded(namespace):
            raise RuntimeError(
                f"Output directory {outdir} is already loaded. "
                f"You can not override a loaded module."
            )
        # clear existing generated files
        if (
            reuse == ReuseType.OVERRIDE
            or not is_graph_config_match
            or not all([meta_file.exists() for meta_file in trace_meta_files])
        ):
            # we have to trace the graph again if not all meta files are present even when reuse=graph.
            print(f"{__name__} | OVERRIDE | Override existing files in {outdir}")
            glob_pattern = "*"
        else:
            print(
                f"{__name__} | GRAPH | keep the graph dump in {outdir} and regenerate the code"
            )
            glob_pattern = "*.py"  # so we can keep graph dumps.
        _clean_files(outdir, glob_pattern)

    return outdir, reusable, transferred


def parallelize(
    module_or_module_class: Union[torch.nn.Module, Type[torch.nn.Module]],
    dummy_forward_args: Dict[str, Any],
    pas_policy: Union[str, Callable[[IRGraph, ComputeConfig], IRGraph]],
    compute_config: ComputeConfig,
    *,
    gen_savedir: Union[str, Path] = "./.nnscaler",
    reuse: Union[ReuseType, str] = ReuseType.MATCH,
    instance_name: Optional[str] = None,
    load_module: bool = True,
    module_dtype: Optional[torch.dtype] = None,
    module_fn: Optional[Callable[[], torch.nn.Module]] = None,
    init_module_params: bool = True,
    broadcast_strategy: Union[str, BroadcastGenFilesStrategy] = "none",
    transfer_config: Optional[Dict[str, Any]] = None,
    force_broadcast_all: bool = False,
) -> Union[None, ParallelModule, Type[ParallelModule]]:
    if isinstance(module_or_module_class, ParallelModule) or (
        inspect.isclass(module_or_module_class)
        and issubclass(module_or_module_class, ParallelModule)
    ):
        # already done
        return module_or_module_class if load_module else None

    if isinstance(module_or_module_class, CubeModule) or (
        inspect.isclass(module_or_module_class)
        and issubclass(module_or_module_class, CubeModule)
    ):
        raise RuntimeError("Old style CubeModule is not supported")

    if isinstance(pas_policy, str):
        if not pas_policy in _CUSTOM_PREDEFINED_POLICIES:
            raise ValueError(f"Invalid pas_policy: {pas_policy}")
        pas_policy = _CUSTOM_PREDEFINED_POLICIES[pas_policy]

    is_module_class = inspect.isclass(module_or_module_class)
    module_class = (
        module_or_module_class if is_module_class else module_or_module_class.__class__
    )
    reuse = ReuseType(reuse) if isinstance(reuse, str) else reuse
    broadcast_strategy = (
        BroadcastGenFilesStrategy(broadcast_strategy)
        if isinstance(broadcast_strategy, str)
        else broadcast_strategy
    )

    # Call it here just to ensure the device group is initialized.
    # If the user initializes dist
    #     and doesn't call `nnscaler.init()` before calling this function, this is necessary.
    if dist.is_initialized():
        _ = DeviceGroup()

    # generate code only in node0
    # if it is not in a torchrun environment, just generate.
    if not dist.is_initialized() or dist.get_rank() == 0:
        outdir, reusable, transferred = _prepare_and_check_reusable(
            gen_savedir,
            module_class,
            compute_config,
            instance_name,
            reuse,
            transfer_config,
        )
        if not reusable:
            config_file = outdir / ParallelModule.COMPUTE_CONFIG_FILE
            ComputeConfig.safe_dump_to_file(
                compute_config, config_file
            )  # always refresh compute config
            with _compile_flags(compute_config):
                regen_status = _gencode(
                    module_or_module_class,
                    dummy_forward_args,
                    pas_policy,
                    compute_config,
                    outdir,
                    module_dtype=module_dtype,
                    module_fn=module_fn,
                )
        else:
            regen_status = RegenStatus.NONE
            logger.info(f"Reuse generated code in {outdir}")

        if regen_status == RegenStatus.CODE and transferred:
            regen_status = RegenStatus.ALL

    if dist.is_initialized():
        # code generation can take very long time (for example, over 1 hour)
        # It is not always OK to use dist.barrier() directly.
        # because the default timeout for nccl is 30 minutes
        # (we can't control the timeout setting if dist is not initialized by us)
        DeviceGroup().long_barrier()

    if broadcast_strategy != BroadcastGenFilesStrategy.NONE or force_broadcast_all:
        if not dist.is_initialized():  # we only support loading in torchrun environment
            raise RuntimeError(
                "Broadcast generated files failed: dist is not initialized."
            )
        dist.barrier()
        # sync regen_status
        curr_rank = dist.get_rank()
        if curr_rank == 0:
            sent_obj = [regen_status]
        else:
            sent_obj = [None]
        dist.broadcast_object_list(
            sent_obj,
            src=0,
        )
        if curr_rank != 0:
            regen_status = sent_obj[0]

        # narrow down broadcast_strategy according to regen_status
        if force_broadcast_all:
            logger.info(f"Force broadcast all generated files in {gen_savedir}")
            broadcast_strategy = BroadcastGenFilesStrategy.ALL
        elif regen_status == RegenStatus.NONE:
            # we don't need to broadcast anything
            broadcast_strategy = BroadcastGenFilesStrategy.NONE
        elif regen_status == RegenStatus.CODE:
            # narrow ALL/NO_WEIGHTS down to code
            broadcast_strategy = BroadcastGenFilesStrategy.CODE
        else:
            # we don't need to narrow broadcast_strategy in this case
            # keep the original broadcast_strategy
            assert regen_status == RegenStatus.ALL

        # broadcast generated files according to regen_status
        if broadcast_strategy != BroadcastGenFilesStrategy.NONE:
            _broadcast_gen_files(
                module_class,
                gen_savedir=gen_savedir,
                instance_name=instance_name,
                broadcast_strategy=broadcast_strategy,
            )

    if load_module:
        if not dist.is_initialized():  # we only support loading in torchrun environment
            raise RuntimeError("Load ParallelModule failed: dist is not initialized.")
        dist.barrier()
        parallel_module_class = _load_parallel_module_class(
            module_class,
            gen_savedir=gen_savedir,
            instance_name=instance_name,
        )
        if is_module_class:
            return parallel_module_class
        else:
            parallel_module = parallel_module_class(init_module_params)
            parallel_module.train(
                module_or_module_class.training
            )  # set training state to the same as original module
            return parallel_module
