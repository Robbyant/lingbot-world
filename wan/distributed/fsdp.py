import gc
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.utils import _free_storage


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
    use_lora=False
):
    # Inference-only optimization: keep parameters gathered after the
    # first forward. FSDP1's equivalent of FSDP2's reshard_after_forward=False
    # is ShardingStrategy.SHARD_GRAD_OP — only grads/optimizer state are
    # sharded; params are unsharded after the first all-gather and stay
    # resident across forwards. Memory cost: ~28GB unsharded params per
    # rank (vs ~3.5GB sharded across 8 ranks), but at 80GB H100 we have
    # headroom. Profiling identified AllGather as 91% of NCCL time
    # (975 ms / 2-chunk window) — skipping the per-forward re-gather
    # across the 35 forwards in a generate() should reclaim most of it.
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states,
        use_orig_params=True if use_lora else False)
    return model


def free_model(model):
    for m in model.modules():
        if isinstance(m, FSDP):
            _free_storage(m._handle.flat_param.data)
    del model
    gc.collect()
    torch.cuda.empty_cache()
