#!/usr/bin/env python3
"""
MVTec AD evaluation with full metrics: I-ROC, I-PR, I-F1max, P-ROC, P-PR, P-F1max, P-PRO.

Supports oneprompt mode (--oneprompt-json) and multi-seed averaging (--oneprompt-seeds).

Usage:
    python scripts/eval_mvtec.py \
        --checkpoint best_model.pth \
        --mvtec-root ~/datasets/mvtec_ad \
        --oneprompt-json eval/MVTec-AD/oneprompt_seed1.json \
        --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import label as connected_components_label
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.MvtecDataset import MvtecDataset
from module import MetaUAS
from utils.meta_utils import read_image_as_tensor, safely_load_state_dict

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


def compute_pro_multi_image(
    preds: List[np.ndarray], masks: List[np.ndarray], num_thresh: int = 200
) -> float:
    """Per-Region Overlap (PRO) metric over multiple images.

    For each connected component in every ground-truth mask, compute overlap with
    thresholded predictions, average across all components, and integrate over thresholds.
    """
    # Collect all component masks and corresponding prediction regions
    all_comp_masks: List[np.ndarray] = []
    all_pred_flat_parts: List[np.ndarray] = []

    for pred, mask in zip(preds, masks):
        mask = mask.astype(np.uint8)
        if mask.sum() == 0:
            continue
        labeled, ncomp = connected_components_label(mask)
        for comp_id in range(1, ncomp + 1):
            comp_mask = (labeled == comp_id)
            all_comp_masks.append(comp_mask.flatten())
            all_pred_flat_parts.append(pred.flatten())

    total_components = len(all_comp_masks)
    if total_components == 0:
        return 0.0

    threshs = np.linspace(0.0, 1.0, num_thresh)
    pro_curve = np.zeros(num_thresh, dtype=np.float64)

    for i, thresh in enumerate(threshs):
        total_overlap = 0.0
        for comp_flat, pred_flat in zip(all_comp_masks, all_pred_flat_parts):
            bin_pred = (pred_flat >= thresh).astype(np.uint8)
            comp_size = comp_flat.sum()
            if comp_size == 0:
                continue
            inter = (bin_pred & comp_flat).sum()
            total_overlap += inter / comp_size
        pro_curve[i] = total_overlap / total_components

    return float(np.trapz(pro_curve, threshs)) / threshs[-1] * 100


def compute_image_metrics(
    scores: np.ndarray, labels: np.ndarray
) -> Tuple[float, float, float]:
    """Compute I-ROC, I-PR, I-F1max given image-level anomaly scores and binary labels."""
    # I-ROC
    i_roc = float(roc_auc_score(labels, scores))

    # I-PR
    precision, recall, _ = precision_recall_curve(labels, scores)
    i_pr = float(auc(recall, precision))

    # I-F1max
    _, _, thresholds_roc = roc_curve(labels, scores)
    f1_scores = np.zeros_like(thresholds_roc)
    for i, th in enumerate(thresholds_roc):
        pred_labels = (scores >= th).astype(int)
        tp = ((pred_labels == 1) & (labels == 1)).sum()
        fp = ((pred_labels == 1) & (labels == 0)).sum()
        fn = ((pred_labels == 0) & (labels == 1)).sum()
        f1_scores[i] = (2 * tp) / max(2 * tp + fp + fn, 1e-12)
    i_f1max = float(f1_scores.max())

    return i_roc, i_pr, i_f1max


def compute_pixel_metrics(
    preds: np.ndarray, masks: np.ndarray
) -> Tuple[float, float, float]:
    """Compute P-ROC, P-PR, P-F1max given flattened predictions and ground-truth masks."""
    # P-ROC
    p_roc = float(roc_auc_score(masks, preds))

    # P-PR
    precision, recall, _ = precision_recall_curve(masks, preds)
    p_pr = float(auc(recall, precision))

    # P-F1max — pixel-level max F1
    thresholds = np.linspace(0.0, 1.0, 200)
    f1s = np.zeros_like(thresholds)
    for i, th in enumerate(thresholds):
        bin_pred = (preds >= th).astype(int)
        tp = ((bin_pred == 1) & (masks == 1)).sum()
        fp = ((bin_pred == 1) & (masks == 0)).sum()
        fn = ((bin_pred == 0) & (masks == 1)).sum()
        f1s[i] = (2 * tp) / max(2 * tp + fp + fn, 1e-12)
    p_f1max = float(f1s.max())

    return p_roc, p_pr, p_f1max


def load_mask(path: Optional[str], target_hw: Tuple[int, int]) -> np.ndarray:
    """Load a ground-truth mask, resized to target_hw (H, W). Returns zeros if no path."""
    if path is None:
        return np.zeros(target_hw, dtype=np.uint8)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(target_hw, dtype=np.uint8)
    mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def evaluate_class(
    model: torch.nn.Module,
    device: torch.device,
    dataset_path: str,
    class_name: str,
    prompt_image: torch.Tensor,
    batch_size: int,
    target_size: Tuple[int, int],
) -> Dict[str, float]:
    """Evaluate all metrics for one MVTec class with a fixed prompt."""
    ds = MvtecDataset(dataset_path, class_name)
    loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=False)

    prompt_4d = prompt_image.unsqueeze(0).to(device)

    all_scores: List[float] = []
    all_labels: List[int] = []
    all_preds_per_image: List[np.ndarray] = []
    all_masks_per_image: List[np.ndarray] = []

    for images, labels, mask_paths in loader:
        test_images = images.to(device)
        prompt_expanded = prompt_4d.repeat(test_images.size(0), 1, 1, 1)

        with torch.no_grad():
            model.eval()
            preds = model(test_images, prompt_expanded)

        b = preds.size(0)
        flat = preds.view(b, -1)
        top10, _ = torch.topk(flat, 10, dim=1)
        scores = top10.float().mean(dim=1).cpu().numpy()

        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())

        pred_np = preds.squeeze(1).cpu().numpy()
        for i in range(b):
            all_preds_per_image.append(pred_np[i])
            mask = load_mask(mask_paths[i], target_size)
            all_masks_per_image.append(mask)

    all_scores_np = np.array(all_scores)
    all_labels_np = np.array(all_labels)
    all_preds_flat = np.concatenate([p.flatten() for p in all_preds_per_image])
    all_masks_flat = np.concatenate([m.flatten() for m in all_masks_per_image])

    i_roc, i_pr, i_f1max = compute_image_metrics(all_scores_np, all_labels_np)
    p_roc, p_pr, p_f1max = compute_pixel_metrics(all_preds_flat, all_masks_flat)

    # Compute PRO over all images (sum overlaps and components across images)
    p_pro = compute_pro_multi_image(all_preds_per_image, all_masks_per_image)

    return {
        "I-ROC": i_roc * 100,
        "I-PR": i_pr * 100,
        "I-F1max": i_f1max * 100,
        "P-ROC": p_roc * 100,
        "P-PR": p_pr * 100,
        "P-F1max": p_f1max * 100,
        "P-PRO": p_pro,
    }


def load_oneprompt_map(oneprompt_json: str) -> Dict[str, dict]:
    with open(oneprompt_json, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser(description="MVTec AD evaluation with full metrics")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--mvtec-root", type=str, default=str(Path.home() / "datasets/mvtec_ad"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--target-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--oneprompt-json", type=str, default=None, help="JSON file with prompt image per class")
    p.add_argument("--oneprompt-dir", type=str, default=None, help="Directory with oneprompt_seed*.json files")
    p.add_argument("--oneprompt-seeds", type=str, default=None, help="Comma-separated seeds for multi-run averaging")
    p.add_argument("--prompt-root", type=str, default=None, help="Root for prompt image paths (defaults to --mvtec-root)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    target_size = (args.target_size, args.target_size)
    prompt_root = Path(args.prompt_root) if args.prompt_root else Path(args.mvtec_root)

    # Build list of (seed_tag, oneprompt_json) pairs
    eval_runs: List[Tuple[str, str]] = []
    if args.oneprompt_seeds and args.oneprompt_dir:
        seeds = [int(s.strip()) for s in args.oneprompt_seeds.split(",") if s.strip()]
        for s in seeds:
            json_path = Path(args.oneprompt_dir) / f"oneprompt_seed{s}.json"
            if json_path.is_file():
                eval_runs.append((f"seed{s}", str(json_path)))
    elif args.oneprompt_json:
        eval_runs.append(("single", args.oneprompt_json))
    else:
        # Random mode: no oneprompt file, use random normal image per class
        eval_runs.append(("random", ""))

    all_run_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for run_tag, json_path in eval_runs:
        # Load model
        model = MetaUAS("efficientnet-b4", "unet", 5, 5, 3, "sa", "cat").to(device)
        safely_load_state_dict(model, args.checkpoint, map_location=device)
        model.eval()

        class_metrics: Dict[str, Dict[str, float]] = {}

        if json_path:
            oneprompt_map = load_oneprompt_map(json_path)

        for cls_name in tqdm(CLASS_NAMES, desc=f"Evaluating [{run_tag}]"):
            if json_path and cls_name in oneprompt_map:
                rel_path = oneprompt_map[cls_name]["filename"]
                prompt_path = str(prompt_root / rel_path)
                prompt_img = read_image_as_tensor(prompt_path)
            else:
                # Random normal image
                ds_tmp = MvtecDataset(args.mvtec_root, cls_name)
                prompt_img = ds_tmp.get_random_normal_image()

            metrics = evaluate_class(
                model, device, args.mvtec_root, cls_name, prompt_img,
                args.batch_size, target_size,
            )
            class_metrics[cls_name] = metrics

        # Compute mean
        mean_metrics: Dict[str, float] = {}
        for key in ["I-ROC", "I-PR", "I-F1max", "P-ROC", "P-PR", "P-F1max", "P-PRO"]:
            mean_metrics[key] = float(np.mean([class_metrics[c][key] for c in CLASS_NAMES]))
        class_metrics["mean"] = mean_metrics

        all_run_results[run_tag] = class_metrics

    # Print results
    header = f"{'Class':<14}" + "".join(f"{m:>10}" for m in ["I-ROC", "I-PR", "I-F1max", "P-ROC", "P-PR", "P-F1max", "P-PRO"])
    print(header)
    print("-" * len(header))

    for run_tag, results in all_run_results.items():
        print(f"\n[{run_tag}]")
        for cls_name in CLASS_NAMES + ["mean"]:
            m = results[cls_name]
            bold = "**" if cls_name == "mean" else "  "
            row = f"{bold}{cls_name:<12}{bold}" + "".join(
                f"{m[k]:>10.1f}" for k in ["I-ROC", "I-PR", "I-F1max", "P-ROC", "P-PR", "P-F1max", "P-PRO"]
            )
            print(row)

    # Multi-run averaging
    if len(all_run_results) > 1:
        print("\n=== Multi-run Average ===")
        avg_results: Dict[str, Dict[str, float]] = {}
        for cls_name in CLASS_NAMES + ["mean"]:
            avg_results[cls_name] = {}
            for key in ["I-ROC", "I-PR", "I-F1max", "P-ROC", "P-PR", "P-F1max", "P-PRO"]:
                values = [all_run_results[tag][cls_name][key] for tag in all_run_results]
                avg = float(np.mean(values))
                std = float(np.std(values))
                avg_results[cls_name][key] = avg
                print(f"{cls_name:<14} {key:>8}: {avg:.1f}±{std:.1f}")


if __name__ == "__main__":
    main()
