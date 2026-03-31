from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Iterator

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str = "nccl") -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    dist.barrier()
    return True, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return tensor
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    if average:
        reduced /= get_world_size()
    return reduced


def sync_bn_if_needed(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if enabled and is_dist_avail_and_initialized():
        return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def ddp_model(model: torch.nn.Module, enabled: bool, local_rank: int) -> torch.nn.Module:
    if enabled and is_dist_avail_and_initialized():
        return torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )
    return model


def maybe_autocast(enabled: bool, dtype: torch.dtype = torch.float16):
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()
