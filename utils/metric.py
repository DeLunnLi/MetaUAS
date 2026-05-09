from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset.MvtecDataset import MvtecDataset

DEFAULT_MVTEC_PATH = "../datasets/mvtec_ad"

CLASS_NAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def evaluate_mvtec(
    model: torch.nn.Module,
    device: torch.device,
    dataset_path: Optional[str] = None,
    normal_image_path: Optional[str] = None,
    batch_size: int = 32,
    seed: Optional[int] = None,
    oneprompt_json: Optional[str] = None,
    prompt_image_root: Optional[str] = None,
) -> Tuple[List[float], float]:
    """Per-class AUROC on MVTec AD. Returns (auc_per_class, mean_auc).

    If oneprompt_json is provided, use fixed prompt images per class instead of random sampling.
    prompt_image_root is prepended to paths in oneprompt_json.
    """
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

    path = dataset_path or os.environ.get("MVTEC_AD_PATH", DEFAULT_MVTEC_PATH)
    prompt_root = Path(prompt_image_root) if prompt_image_root else Path(path)

    oneprompt_map: dict[str, str] = {}
    if oneprompt_json:
        with open(oneprompt_json, "r", encoding="utf-8") as f:
            oneprompt_map = json.load(f)

    auc_list: List[float] = []

    for class_name in CLASS_NAMES:
        if class_name in oneprompt_map:
            prompt_path = str(prompt_root / oneprompt_map[class_name])
            ds = MvtecDataset(path, class_name, normal_image_override=prompt_path)
        else:
            ds = MvtecDataset(path, class_name, normal_image_override=normal_image_path)

        image_normal = ds.get_random_normal_image()
        loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=False)
        image_normal = image_normal.unsqueeze(0).to(device)
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels, _mask_paths in loader:
                test_image = images.to(device)
                test_label = labels.to(device)
                image_normal_expanded = image_normal.repeat(test_image.size(0), 1, 1, 1)
                model.eval()
                predict_mask = model(test_image, image_normal_expanded)
                top_100_values, _ = torch.topk(predict_mask.view(test_image.size(0), -1), 10, dim=1)
                mean_top_100 = top_100_values.float().mean(dim=1)
                all_labels.append(test_label.cpu().detach().numpy())
                all_predictions.append(mean_top_100.cpu().detach().numpy())

        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)
        auc_list.append(float(roc_auc_score(all_labels, all_predictions)))

    return auc_list, float(np.mean(auc_list))


def metric(model, device):
    """Legacy interface: returns per-class AUROC list only."""
    aucs, _ = evaluate_mvtec(model, device)
    return aucs
