"""
从 CocoMetaDataset + coco_dataset_split 构建训练用 DataLoader（单机 shuffle 或 DDP + DistributedSampler）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union

import torch
from torch.utils.data import DataLoader, DistributedSampler

from .coco_dataset import DEFAULT_COCO_META_PATHS, CocoMetaDataPaths, CocoMetaDataset
from .coco_dataset_split import load_split


def load_stem_allowlist_file(path: Optional[str]) -> Optional[Set[str]]:
    """每行一个 stem 或带扩展名的文件名，``#`` 开头为注释。"""
    if not path:
        return None
    stems: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stems.add(Path(line).stem)
    return stems if stems else None


def build_metadataset(
    target_size: Union[int, Tuple[int, int]] = (256, 256),
    *,
    split_images_dir: str,
    train_rate: float = 0.95,
    path_to_split: Optional[str] = None,
    paths: Optional[CocoMetaDataPaths] = None,
    split_key: str = "train",
    image_stem_allowlist: Optional[Set[str]] = None,
) -> Tuple[CocoMetaDataset, Dict[str, Any]]:
    """
    :param split_images_dir: 与训练脚本中 path_to_coco 一致，用于列举图片文件名并划分 train/val
    :param paths: COCO 相关根目录 + DTD；默认使用 ``DEFAULT_COCO_META_PATHS``
    :param split_key: "train" 或 "val"
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    split = load_split(split_images_dir, path_to_split=path_to_split, train_rate=train_rate)
    names = split[split_key]
    ds = CocoMetaDataset(
        names,
        target_size,
        paths=paths if paths is not None else DEFAULT_COCO_META_PATHS,
        image_stem_allowlist=image_stem_allowlist,
    )
    return ds, split


def build_train_dataloader(
    dataset: CocoMetaDataset,
    *,
    batch_size: int = 128,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 0,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """
    DDP 时每个 epoch 需调用 ``sampler.set_epoch(epoch)``，否则各卡 shuffle 不刷新。
    """
    sampler: Optional[DistributedSampler] = None
    shuffle = True
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, sampler


def build_meta_training_loader(
    *,
    split_images_dir: str,
    target_size: Union[int, Tuple[int, int]] = (256, 256),
    train_rate: float = 0.95,
    path_to_split: Optional[str] = None,
    paths: Optional[CocoMetaDataPaths] = None,
    split_key: str = "train",
    batch_size: int = 128,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 0,
    stem_allowlist_file: Optional[str] = None,
) -> Tuple[DataLoader, Optional[DistributedSampler], CocoMetaDataset, Dict[str, Any]]:
    """一步构建：CocoMetaDataset + DataLoader（与 train.py / train_metauas.py 中用法等价）。"""
    allow = load_stem_allowlist_file(stem_allowlist_file)
    ds, split = build_metadataset(
        target_size,
        split_images_dir=split_images_dir,
        train_rate=train_rate,
        path_to_split=path_to_split,
        paths=paths,
        split_key=split_key,
        image_stem_allowlist=allow,
    )
    loader, sampler = build_train_dataloader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        seed=seed,
    )
    return loader, sampler, ds, split
