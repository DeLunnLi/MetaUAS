#!/usr/bin/env python3
"""
MVTec AD / VisA evaluation with full metrics: I-ROC, I-PR, I-F1max, P-ROC, P-PR, P-F1max, P-PRO.

Two modes:
  --mode oneprompt : One fixed prompt per class, multi-seed averaging
  --mode topk      : Per-image top-k similar prompts from precomputed pairs

Usage:
    python eval/eval_mvtec.py --dataset mvtec --mode oneprompt --checkpoint model.pth ...
    python eval/eval_mvtec.py --dataset mvtec --mode topk --checkpoint model.pth ...
"""

from __future__ import annotations

import argparse
import json
import os
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
from dataset.VisaDataset import VISA_CLASS_NAMES, VisaDataset
from module import MetaUAS
from utils.meta_utils import read_image_as_tensor, safely_load_state_dict

METRIC_KEYS = ["I-ROC", "I-PR", "I-F1max", "P-ROC", "P-PR", "P-F1max", "P-PRO"]

MVTEC_CLASS_NAMES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]

DATASET_CONFIG = {
    "mvtec": {"class_names": MVTEC_CLASS_NAMES, "dataset_cls": MvtecDataset},
    "visa": {"class_names": VISA_CLASS_NAMES, "dataset_cls": VisaDataset},
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_pro_multi_image(
    preds: List[np.ndarray], masks: List[np.ndarray], num_thresh: int = 200
) -> float:
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


def compute_image_metrics(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    i_roc = float(roc_auc_score(labels, scores))
    precision, recall, _ = precision_recall_curve(labels, scores)
    i_pr = float(auc(recall, precision))
    _, _, thresholds_roc = roc_curve(labels, scores)
    f1_scores = np.zeros_like(thresholds_roc)
    for i, th in enumerate(thresholds_roc):
        pred_labels = (scores >= th).astype(int)
        tp = ((pred_labels == 1) & (labels == 1)).sum()
        fp = ((pred_labels == 1) & (labels == 0)).sum()
        fn = ((pred_labels == 0) & (labels == 1)).sum()
        f1_scores[i] = (2 * tp) / max(2 * tp + fp + fn, 1e-12)
    return i_roc, i_pr, float(f1_scores.max())


def compute_pixel_metrics(preds: np.ndarray, masks: np.ndarray) -> Tuple[float, float, float]:
    p_roc = float(roc_auc_score(masks, preds))
    precision, recall, _ = precision_recall_curve(masks, preds)
    p_pr = float(auc(recall, precision))
    thresholds = np.linspace(0.0, 1.0, 200)
    f1s = np.zeros_like(thresholds)
    for i, th in enumerate(thresholds):
        bin_pred = (preds >= th).astype(int)
        tp = ((bin_pred == 1) & (masks == 1)).sum()
        fp = ((bin_pred == 1) & (masks == 0)).sum()
        fn = ((bin_pred == 0) & (masks == 1)).sum()
        f1s[i] = (2 * tp) / max(2 * tp + fp + fn, 1e-12)
    return p_roc, p_pr, float(f1s.max())


def load_mask(path: Optional[str], target_hw: Tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(target_hw, dtype=np.uint8)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(target_hw, dtype=np.uint8)
    mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_batch(model: torch.nn.Module, images: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
    """Returns sigmoid mask (B, H, W)."""
    model.eval()
    return model(images, prompts).squeeze(1)


def collect_metrics(
    all_scores: List[float],
    all_labels: List[int],
    all_preds_per_image: List[np.ndarray],
    all_masks_per_image: List[np.ndarray],
) -> Dict[str, float]:
    scores_np = np.array(all_scores)
    labels_np = np.array(all_labels)
    preds_flat = np.concatenate([p.flatten() for p in all_preds_per_image])
    masks_flat = np.concatenate([m.flatten() for m in all_masks_per_image])

    i_roc, i_pr, i_f1max = compute_image_metrics(scores_np, labels_np)
    p_roc, p_pr, p_f1max = compute_pixel_metrics(preds_flat, masks_flat)
    p_pro = compute_pro_multi_image(all_preds_per_image, all_masks_per_image)

    return {
        "I-ROC": i_roc * 100, "I-PR": i_pr * 100, "I-F1max": i_f1max * 100,
        "P-ROC": p_roc * 100, "P-PR": p_pr * 100, "P-F1max": p_f1max * 100,
        "P-PRO": p_pro,
    }


# ---------------------------------------------------------------------------
# Mode: oneprompt
# ---------------------------------------------------------------------------

def evaluate_class_oneprompt(
    model: torch.nn.Module,
    device: torch.device,
    dataset_path: str,
    class_name: str,
    prompt_image: torch.Tensor,
    batch_size: int,
    target_size: Tuple[int, int],
    dataset_cls= MvtecDataset,
) -> Dict[str, float]:
    ds = dataset_cls(dataset_path, class_name)
    loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=False)
    prompt_4d = prompt_image.unsqueeze(0).to(device)

    all_scores: List[float] = []
    all_labels: List[int] = []
    all_preds: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []

    for images, labels, mask_paths in loader:
        test_images = images.to(device)
        prompts = prompt_4d.repeat(test_images.size(0), 1, 1, 1)
        preds = predict_batch(model, test_images, prompts)

        b = preds.size(0)
        flat = preds.view(b, -1)
        top10, _ = torch.topk(flat, 10, dim=1)
        scores = top10.float().mean(dim=1).cpu().numpy()
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())

        pred_np = preds.cpu().numpy()
        for i in range(b):
            all_preds.append(pred_np[i])
            all_masks.append(load_mask(mask_paths[i], target_size))

    return collect_metrics(all_scores, all_labels, all_preds, all_masks)


def run_oneprompt(args, device, target_size, data_root, prompt_root, class_names, dataset_cls):
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
        eval_runs.append(("random", ""))

    all_run_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for run_tag, json_path in eval_runs:
        model = MetaUAS("efficientnet-b4", "unet", 5, 5, 3, "sa", "cat").to(device)
        safely_load_state_dict(model, args.checkpoint, map_location=device)

        prompt_map = {}
        if json_path:
            with open(json_path, "r", encoding="utf-8") as f:
                prompt_map = json.load(f)

        class_metrics: Dict[str, Dict[str, float]] = {}
        for cls_name in tqdm(class_names, desc=f"Oneprompt [{run_tag}]"):
            if cls_name in prompt_map:
                rel = prompt_map[cls_name]["filename"]
                prompt_img = read_image_as_tensor(str(prompt_root / rel))
            else:
                ds_tmp = dataset_cls(data_root, cls_name)
                prompt_img = ds_tmp.get_random_normal_image()

            class_metrics[cls_name] = evaluate_class_oneprompt(
                model, device, data_root, cls_name, prompt_img,
                args.batch_size, target_size, dataset_cls,
            )

        mean = {k: float(np.mean([class_metrics[c][k] for c in class_names])) for k in METRIC_KEYS}
        class_metrics["mean"] = mean
        all_run_results[run_tag] = class_metrics

    return all_run_results, class_names


# ---------------------------------------------------------------------------
# Mode: topk
# ---------------------------------------------------------------------------

def load_topk_pairs(jsonl_path: str) -> Dict[str, List[str]]:
    """Load test-train-top10pair JSONL. Returns {test_image_relpath: [prompt_relpath, ...]}."""
    pairs: Dict[str, List[str]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for test_path, prompt_list in obj.items():
                pairs[test_path] = prompt_list
    return pairs


def evaluate_class_topk(
    model: torch.nn.Module,
    device: torch.device,
    dataset_path: str,
    class_name: str,
    topk_pairs: Dict[str, List[str]],
    prompt_root: Path,
    batch_size: int,
    target_size: Tuple[int, int],
    dataset_cls= MvtecDataset,
    top_k: int = 1,
) -> Dict[str, float]:
    ds = dataset_cls(dataset_path, class_name)

    all_scores: List[float] = []
    all_labels: List[int] = []
    all_preds: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []

    for idx in tqdm(range(len(ds)), desc=f"  {class_name}", leave=False):
        image, label, mask_path = ds[idx]
        test_img = image.unsqueeze(0).to(device)

        # Find matching prompts for this test image
        img_path = ds.test_img_paths[idx]
        # Normalize path to match JSON keys (relative to dataset root)
        rel_test = os.path.relpath(img_path, dataset_path) if img_path.startswith(dataset_path) else img_path
        prompts = topk_pairs.get(rel_test, [])

        if not prompts:
            # Fallback: use a random normal image
            prompt_img = ds.get_random_normal_image().unsqueeze(0).to(device)
            pred = predict_batch(model, test_img, prompt_img)[0].cpu().numpy()
        else:
            k = min(top_k, len(prompts))
            preds_k = []
            for pi in range(k):
                prompt_path = str(prompt_root / prompts[pi])
                prompt_img = read_image_as_tensor(prompt_path).unsqueeze(0).to(device)
                pred = predict_batch(model, test_img, prompt_img)[0].cpu().numpy()
                preds_k.append(pred)
            pred = np.mean(preds_k, axis=0)

        # Image-level score: mean of top-10 pixels
        flat = pred.flatten()
        top10 = np.sort(flat)[-10:]
        score = float(top10.mean())
        all_scores.append(score)
        all_labels.append(label)

        all_preds.append(pred)
        all_masks.append(load_mask(mask_path, target_size))

    return collect_metrics(all_scores, all_labels, all_preds, all_masks)


def run_topk(args, device, target_size, data_root, prompt_root, class_names, dataset_cls):
    topk_path = args.topk_json
    if not Path(topk_path).is_file():
        raise FileNotFoundError(f"TopK pairs file not found: {topk_path}")

    pairs = load_topk_pairs(topk_path)
    print(f"Loaded {len(pairs)} test-image -> prompt-list entries")

    model = MetaUAS("efficientnet-b4", "unet", 5, 5, 3, "sa", "cat").to(device)
    safely_load_state_dict(model, args.checkpoint, map_location=device)

    class_metrics: Dict[str, Dict[str, float]] = {}
    for cls_name in class_names:
        class_metrics[cls_name] = evaluate_class_topk(
            model, device, data_root, cls_name, pairs, prompt_root,
            args.batch_size, target_size, dataset_cls, args.top_k,
        )

    mean = {k: float(np.mean([class_metrics[c][k] for c in class_names])) for k in METRIC_KEYS}
    class_metrics["mean"] = mean

    return {f"top{args.top_k}": class_metrics}, class_names


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_markdown_table(all_run_results: dict, class_names: list):
    """Print results in a markdown-style table for each run and the average."""
    for run_tag, results in all_run_results.items():
        print(f"\n### {run_tag}")
        header = f"| {'Class':<14} |" + "|".join(f" {m:>10} " for m in METRIC_KEYS) + "|"
        sep = "|" + "|".join(":---:" for _ in range(len(METRIC_KEYS) + 1)) + "|"
        print(header)
        print(sep)
        for cls_name in class_names + ["mean"]:
            m = results[cls_name]
            bold = "**" if cls_name == "mean" else ""
            row = f"| {bold}{cls_name:<12}{bold} |" + "|".join(f" {m[k]:>10.1f} " for k in METRIC_KEYS) + "|"
            print(row)

    if len(all_run_results) > 1:
        print("\n### Average (mean±std)")
        for cls_name in class_names + ["mean"]:
            parts = []
            for k in METRIC_KEYS:
                vals = [all_run_results[tag][cls_name][k] for tag in all_run_results]
                parts.append(f"{np.mean(vals):.1f}±{np.std(vals):.1f}")
            print(f"{cls_name:<14} " + "  ".join(f"{p:>12}" for p in parts))


def print_plain_table(all_run_results: dict, class_names: list):
    """Print results in a plain-text aligned table."""
    header = f"{'Class':<14}" + "".join(f"{m:>10}" for m in METRIC_KEYS)
    print(header)
    print("-" * len(header))
    for run_tag, results in all_run_results.items():
        print(f"\n[{run_tag}]")
        for cls_name in class_names + ["mean"]:
            m = results[cls_name]
            bold = "**" if cls_name == "mean" else "  "
            row = f"{bold}{cls_name:<12}{bold}" + "".join(f"{m[k]:>10.1f}" for k in METRIC_KEYS)
            print(row)
    if len(all_run_results) > 1:
        print("\n=== Multi-run Average (mean±std) ===")
        for cls_name in class_names + ["mean"]:
            parts = []
            for k in METRIC_KEYS:
                vals = [all_run_results[tag][cls_name][k] for tag in all_run_results]
                parts.append(f"{np.mean(vals):.1f}±{np.std(vals):.1f}")
            print(f"{cls_name:<14}" + "".join(f"{p:>10}" for p in parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MVTec AD / VisA evaluation")
    p.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    p.add_argument("--mode", type=str, default="oneprompt", choices=["oneprompt", "topk"])
    p.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint (.pth)")
    p.add_argument("--data-root", type=str, default=None, help="Dataset root")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--target-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=None)
    # oneprompt args
    p.add_argument("--oneprompt-json", type=str, default=None)
    p.add_argument("--oneprompt-dir", type=str, default=None)
    p.add_argument("--oneprompt-seeds", type=str, default=None)
    p.add_argument("--prompt-root", type=str, default=None)
    # topk args
    p.add_argument("--topk-json", type=str, default=None, help="Path to test-train-top10pair JSONL")
    p.add_argument("--top-k", type=int, default=1, help="Number of top similar prompts to average")
    # output format
    p.add_argument("--markdown", action="store_true", help="Print results as markdown table")
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    target_size = (args.target_size, args.target_size)
    ds_cfg = DATASET_CONFIG[args.dataset]
    class_names = ds_cfg["class_names"]
    dataset_cls = ds_cfg["dataset_cls"]

    if args.data_root:
        data_root = args.data_root
    elif args.dataset == "visa":
        data_root = str(Path.home() / "datasets/visa")
    else:
        data_root = str(Path.home() / "datasets/mvtec_ad")
    prompt_root = Path(args.prompt_root) if args.prompt_root else Path(data_root)

    if args.oneprompt_dir is None and args.mode == "oneprompt":
        args.oneprompt_dir = f"eval/{'VisA-AD' if args.dataset == 'visa' else 'MVTec-AD'}"
    if args.topk_json is None and args.mode == "topk":
        args.topk_json = f"eval/{'VisA-AD' if args.dataset == 'visa' else 'MVTec-AD'}/test-train-top10pair-eb4.json"

    if args.mode == "oneprompt":
        all_run_results, class_names_out = run_oneprompt(
            args, device, target_size, data_root, prompt_root, class_names, dataset_cls,
        )
    else:
        all_run_results, class_names_out = run_topk(
            args, device, target_size, data_root, prompt_root, class_names, dataset_cls,
        )

    if args.markdown:
        print_markdown_table(all_run_results, class_names_out)
    else:
        print_plain_table(all_run_results, class_names_out)


if __name__ == "__main__":
    main()
