"""DDP setup/teardown and shared CocoMetaDataset + DataLoader utilities."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from dataset.coco_dataset import DEFAULT_COCO_META_PATHS, CocoMetaDataPaths, CocoMetaDataset
from dataset.coco_dataset_split import load_split
from dataset.dataloader import load_stem_allowlist_file

def ddp_setup(backend: str = "nccl") -> Tuple[int, int, int, torch.device]:
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, global_rank, world_size, device


def ddp_teardown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def make_metadataset_train_loader(
    path_to_coco: str,
    train_rate: float,
    target_size: Union[int, Tuple[int, int]],
    batch_size: int,
    world_size: int,
    local_rank: int,
    num_workers: int = 8,
    pin_memory: bool = True,
    path_to_split: Optional[str] = None,
    paths: Optional[CocoMetaDataPaths] = None,
    stem_allowlist_file: Optional[str] = None,
    aug_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], CocoMetaDataset, DistributedSampler, DataLoader]:
    allow = load_stem_allowlist_file(stem_allowlist_file)
    dataset_split = load_split(
        path_to_coco, path_to_split=path_to_split, train_rate=train_rate, stem_allowlist=allow
    )
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    p = paths if paths is not None else DEFAULT_COCO_META_PATHS
    # Allowlist already applied during split; skip second filtering here
    dataset = CocoMetaDataset(
        dataset_split["train"],
        target_size,
        paths=p,
        image_stem_allowlist=None,
        aug_kwargs=aug_kwargs,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset_split, dataset, sampler, loader


def make_metadataset_val_loader(
    path_to_coco: str,
    train_rate: float,
    target_size: Union[int, Tuple[int, int]],
    batch_size: int,
    world_size: int,
    local_rank: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    path_to_split: Optional[str] = None,
    paths: Optional[CocoMetaDataPaths] = None,
    aug_kwargs: Optional[Dict[str, Any]] = None,
    stem_allowlist_file: Optional[str] = None,
) -> Tuple[CocoMetaDataset, DistributedSampler, DataLoader]:
    """Validation loader using the same split as training; returns val subset."""
    allow = load_stem_allowlist_file(stem_allowlist_file)
    dataset_split = load_split(
        path_to_coco, path_to_split=path_to_split, train_rate=train_rate, stem_allowlist=allow
    )
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    p = paths if paths is not None else DEFAULT_COCO_META_PATHS
    dataset = CocoMetaDataset(
        dataset_split["val"],
        target_size,
        paths=p,
        image_stem_allowlist=None,
        aug_kwargs=aug_kwargs,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, sampler, loader


def all_reduce_mean_epoch_loss(epoch_loss_sum: float, num_batches: int, device: torch.device) -> float:
    """Global average loss via all_reduce sum of per-rank loss totals over total batches."""
    t = torch.tensor([epoch_loss_sum, float(num_batches)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    total_loss, total_n = t[0].item(), max(int(t[1].item()), 1)
    return total_loss / total_n


@torch.no_grad()
def eval_meta_model_avg_loss(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    world_size: int,
) -> float:
    """Distributed validation BCE — same model(query, prompt) as training, mask aligned to query."""
    model.eval()
    tot, n = 0.0, 0
    for prompt, query, mask in dataloader:
        prompt = prompt.to(device, non_blocking=True)
        query = query.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pred = model(query, prompt)
        loss = loss_fn(pred, mask.clamp(0.0, 1.0))
        bs = query.size(0)
        tot += loss.item() * bs
        n += bs
    t = torch.tensor([tot, float(n)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / max(t[1], 1.0)).item()
