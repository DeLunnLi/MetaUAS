#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MetaUAS 辅助：随机种子、读图、权重加载、分数图可视化等（原 ``module/meta_utils``）。"""

from __future__ import annotations

import os
import random
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def set_random_seed(seed: int = 233, reproduce: bool = False) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed**2)
    torch.cuda.manual_seed(seed**3)
    random.seed(seed**4)
    if reproduce:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def normalize(
    pred: np.ndarray,
    max_value: Optional[float] = None,
    min_value: Optional[float] = None,
) -> np.ndarray:
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return (pred - min_value) / (max_value - min_value + 1e-8)


def apply_ad_scoremap(image: np.ndarray, scoremap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    np_image = np.asarray(image, dtype=np.float32)
    scoremap_u8 = (scoremap * 255).astype(np.uint8)
    scoremap_u8 = cv2.applyColorMap(scoremap_u8, cv2.COLORMAP_JET)
    scoremap_u8 = cv2.cvtColor(scoremap_u8, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap_u8).astype(np.uint8)


def read_image_as_tensor(path_to_image: Union[str, os.PathLike]) -> torch.Tensor:
    pil_image = Image.open(path_to_image).convert("RGB")
    return pil_to_tensor(pil_image).float() / 255.0


def safely_load_state_dict(
    model: nn.Module,
    checkpoint: Union[str, os.PathLike],
    map_location: Optional[Any] = None,
    strict: bool = True,
) -> nn.Module:
    """Load weights; supports full checkpoints and raw state_dict (PyTorch 2.6+ safe)."""
    load_kw: dict = {}
    if map_location is not None:
        load_kw["map_location"] = map_location
    try:
        load_kw["weights_only"] = False
        obj = torch.load(checkpoint, **load_kw)
    except TypeError:
        load_kw.pop("weights_only", None)
        obj = torch.load(checkpoint, **load_kw)

    if isinstance(obj, dict):
        if "state_dict" in obj:
            obj = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            obj = obj["model"]
    model.load_state_dict(obj, strict=strict)
    return model
