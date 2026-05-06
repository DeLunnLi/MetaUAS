#!/usr/bin/env python3
"""
MetaUAS DDP training: CYWS coco-inpainted pipeline (data_split.pkl train/val split).

- object-level: split between COCO instance paste exchange and CYWS inpaint pairs by exchange_p
- CYWS inpaint: random two-view, mask XOR for change region, unified geometric augmentation
- DataLoader returns (prompt, query, mask), mask aligned to query

Usage: torchrun --nproc_per_node=4 train/train_change_pairs.py --coco-inpainted-root /path/to/coco-inpainted/train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure top-level packages (dataset/ module/ utils/) are importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LambdaLR
from tqdm import tqdm

from dataset.coco_inpainted_dataset import (
    _scan_coco_images_for_exchange,
    make_coco_inpainted_train_loader,
    make_coco_inpainted_val_loader,
)
from module import MetaUAS
from utils.metric import evaluate_mvtec
from train_ddp_common import (
    all_reduce_mean_epoch_loss,
    ddp_setup,
    ddp_teardown,
    eval_meta_model_avg_loss,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="MetaUAS DDP: CYWS coco-inpainted data (see arXiv:2505.09265)"
    )
    p.add_argument("--target-size", type=int, default=256)
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Per-GPU batch size (DDP); B4+dual-image+cat may OOM at 128 on 24GB, reduce to 16 if needed",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Defaults to 30, matching the paper (arXiv:2505.09265)",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        dest="weight_decay",
        help="AdamW weight decay; paper uses 0.0005",
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--scheduler",
        type=str,
        choices=("exponential", "cosine", "constant"),
        default="cosine",
        help="constant: fixed learning rate throughout training",
    )
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--eta-min", type=float, default=1e-6)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--best-path", type=str, default="best_model_change_pairs.pth")
    p.add_argument("--no-epoch-checkpoints", action="store_true")
    p.add_argument("--checkpoint-prefix", type=str, default="metauas_change_pairs_epoch")
    p.add_argument(
        "--val-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run validation after each epoch (disable: --no-val-each-epoch); requires all_reduce across ranks",
    )
    p.add_argument(
        "--val-batch-size",
        type=int,
        default=32,
        help="Per-GPU validation batch size; can differ from training to save memory",
    )

    # ========= MVTec eval each epoch (rank0 only) =========
    p.add_argument(
        "--mvtec-eval-each-epoch",
        action="store_true",
        help="Evaluate on MVTec after each epoch (rank0 only), save best ckpt by mean_auc",
    )
    p.add_argument(
        "--mvtec-root",
        type=str,
        default=str(Path.home() / "datasets/mvtec_ad"),
        help="MVTec AD root directory (contains bottle/ cable/ ...)",
    )
    p.add_argument(
        "--mvtec-device",
        type=str,
        default="local",
        help="Device for MVTec eval: 'local' uses each rank's training GPU (recommended), or explicit e.g. cuda:0",
    )
    p.add_argument("--mvtec-batch-size", type=int, default=32)
    p.add_argument(
        "--mvtec-seed",
        type=int,
        default=1,
        help="Base seed for MVTec eval; with multi-GPU avg, each rank uses seed+global_rank",
    )
    p.add_argument(
        "--mvtec-oneprompt-json",
        type=str,
        default="",
        help="If provided: use oneprompt_seed*.json for fixed per-class prompts (overrides random seed sampling)",
    )
    p.add_argument(
        "--mvtec-oneprompt-seeds",
        type=str,
        default="",
        help=(
            "Comma-separated oneprompt seeds per rank for multi-GPU eval (e.g. '1,2,3,5'). "
            "When set with --mvtec-oneprompt-json/--mvtec-oneprompt-dir, rank i uses seeds[i]."
        ),
    )
    p.add_argument(
        "--mvtec-oneprompt-dir",
        type=str,
        default="",
        help="Directory containing oneprompt_seed*.json (optional; inferred from --mvtec-oneprompt-json if empty)",
    )
    p.add_argument(
        "--mvtec-prompt-root",
        type=str,
        default="",
        help="Root directory for filenames in oneprompt json; defaults to --mvtec-root if empty",
    )
    p.add_argument(
        "--best-mvtec-path",
        type=str,
        default="best_model_mvtec_auc.pth",
        help="Best ckpt saved by MVTec mean_auc (written by rank0)",
    )

    # ========= Augmentations (CYWS-style, controlled via CLI) =========
    p.add_argument("--aug-crop-p", type=float, default=0.0)
    p.add_argument(
        "--aug-crop-independent",
        action="store_true",
        help="If set: independently crop two images and strictly align masks via CYWS intersection logic (harder)",
    )
    p.add_argument("--aug-crop-scale-min", type=float, default=0.92)
    p.add_argument("--aug-crop-scale-max", type=float, default=1.0)
    p.add_argument("--aug-crop-ratio-min", type=float, default=0.95)
    p.add_argument("--aug-crop-ratio-max", type=float, default=1.05)
    p.add_argument("--aug-hflip-p", type=float, default=0.0)
    p.add_argument("--aug-vflip-p", type=float, default=0.0)
    p.add_argument("--aug-affine-degrees", type=float, default=30.0)
    p.add_argument("--aug-affine-translate", type=float, default=0.2)
    p.add_argument("--aug-affine-scale-min", type=float, default=0.8)
    p.add_argument("--aug-affine-scale-max", type=float, default=1.5)
    p.add_argument("--aug-affine-p", type=float, default=1.0)
    p.add_argument("--aug-jitter-b", type=float, default=0.1)
    p.add_argument("--aug-jitter-c", type=float, default=0.1)
    p.add_argument("--aug-jitter-s", type=float, default=0.1)
    p.add_argument("--aug-jitter-h", type=float, default=0.1)
    p.add_argument("--aug-jitter-p", type=float, default=1.0)
    p.add_argument(
        "--coco-inpainted-root",
        type=str,
        default="/home/ldl/metauas/coco-inpainted/train",
        help="Dataset root directory (contains data_split.pkl, images_and_masks, inpainted)",
    )
    p.add_argument(
        "--coco-inpainted-local-region-p",
        type=float,
        default=0.5,
        help="Probability of local-region (DRAEM/Perlin) synthesis; rest is object-level (CYWS inpaint / exchange)",
    )
    p.add_argument(
        "--coco-inpainted-dtd-root",
        type=str,
        default="/home/ldl/datasets/dtd/images",
        help="Texture source root for local-region branch (DTD)",
    )
    p.add_argument(
        "--coco-inpainted-exchange-p",
        type=float,
        default=0.5,
        help="Within object-level: probability of COCO instance paste exchange; otherwise CYWS inpaint (needs valid COCO images+meta)",
    )
    p.add_argument(
        "--coco-inpainted-coco-images",
        type=str,
        default="/home/ldl/datasets/images/train2017",
        help="Exchange source: MS-COCO train image directory (*.jpg etc.)",
    )
    p.add_argument(
        "--coco-inpainted-coco-metadata",
        type=str,
        default="/home/ldl/datasets/images/meta_data",
        help="Exchange source: directory of *.npy instance annotations (same as pre_process_coco)",
    )
    p.add_argument(
        "--coco-inpainted-exchange-max-area",
        type=float,
        default=0.35,
        help="Maximum area ratio for exchange paste region (relative to full image)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    local_rank, global_rank, world_size, device = ddp_setup()

    aug_kwargs = dict(
        crop_p=args.aug_crop_p,
        crop_independent=args.aug_crop_independent,
        crop_scale=(args.aug_crop_scale_min, args.aug_crop_scale_max),
        crop_ratio=(args.aug_crop_ratio_min, args.aug_crop_ratio_max),
        hflip_p=args.aug_hflip_p,
        vflip_p=args.aug_vflip_p,
        affine_degrees=args.aug_affine_degrees,
        affine_translate=(args.aug_affine_translate, args.aug_affine_translate),
        affine_scale=(
            args.aug_affine_scale_min,
            args.aug_affine_scale_max,
            args.aug_affine_scale_min,
            args.aug_affine_scale_max,
        ),
        affine_p=args.aug_affine_p,
        color_jitter=(args.aug_jitter_b, args.aug_jitter_c, args.aug_jitter_s, args.aug_jitter_h),
        color_jitter_p=args.aug_jitter_p,
    )

    model = MetaUAS(
        "efficientnet-b4",
        "unet",
        5,
        5,
        3,
        "sa",
        "cat",
    ).to(device)
    # UnetDecoder with skip=None bypasses SCSE attention1; some params may have no grad, need find_unused_parameters
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # Encoder is frozen; only optimize alignment / decoder / segmentation head
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=float(args.weight_decay),
    )

    if args.scheduler == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == "constant":
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    ci_root = args.coco_inpainted_root.strip()
    if not ci_root:
        raise ValueError("--coco-inpainted-root must not be empty")

    if float(args.coco_inpainted_exchange_p) > 0.0:
        ex_img = Path(str(args.coco_inpainted_coco_images).strip())
        ex_meta = Path(str(args.coco_inpainted_coco_metadata).strip())
        if (not ex_img.is_dir()) or (not ex_meta.is_dir()) or (not _scan_coco_images_for_exchange(ex_img)):
            if global_rank == 0:
                print(
                    "[train_change_pairs] exchange disabled (invalid COCO image dir or meta_data, "
                    "or no images found). Only local-region + CYWS inpaint.",
                    flush=True,
                )
            args.coco_inpainted_exchange_p = 0.0

    dataset_split, train_dataset, train_sampler, dataloader = make_coco_inpainted_train_loader(
        ci_root,
        args.target_size,
        args.batch_size,
        world_size,
        local_rank,
        num_workers=args.num_workers,
        aug_kwargs=aug_kwargs,
        local_region_p=float(args.coco_inpainted_local_region_p),
        dtd_root=str(args.coco_inpainted_dtd_root),
        exchange_p=float(args.coco_inpainted_exchange_p),
        coco_exchange_images_dir=str(args.coco_inpainted_coco_images).strip(),
        coco_exchange_metadata_dir=str(args.coco_inpainted_coco_metadata).strip(),
        exchange_max_area_ratio=float(args.coco_inpainted_exchange_max_area),
    )

    val_loader = None
    val_sampler = None
    if args.val_each_epoch:
        _, val_sampler, val_loader = make_coco_inpainted_val_loader(
            args.coco_inpainted_root.strip(),
            args.target_size,
            args.val_batch_size,
            world_size,
            local_rank,
            num_workers=args.num_workers,
            aug_kwargs=aug_kwargs,
            local_region_p=float(args.coco_inpainted_local_region_p),
            dtd_root=str(args.coco_inpainted_dtd_root),
            exchange_p=float(args.coco_inpainted_exchange_p),
            coco_exchange_images_dir=str(args.coco_inpainted_coco_images).strip(),
            coco_exchange_metadata_dir=str(args.coco_inpainted_coco_metadata).strip(),
            exchange_max_area_ratio=float(args.coco_inpainted_exchange_max_area),
        )

    if global_rank == 0:
        n_tr = sum(p.numel() for p in trainable)
        n_fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(
            f"[train_change_pairs] coco-inpainted root={args.coco_inpainted_root!r} "
            f"| pairs_train={dataset_split.get('num_pairs', '?')} (split=data_split.pkl) "
            f"| local_p={args.coco_inpainted_local_region_p} exchange_p={args.coco_inpainted_exchange_p} "
            f"| mask_align=query "
            f"| cyws_inpaint=random_two_views "
            f"| coco_ex_images={args.coco_inpainted_coco_images!r}"
        )
        print(
            f"[train_change_pairs] encoder frozen | AdamW trainable {n_tr:,} / frozen {n_fr:,} "
            f"| lr={args.lr} weight_decay={args.weight_decay}"
        )
        print(
            f"[train_change_pairs] val_each_epoch={args.val_each_epoch} | best_path={args.best_path!r}"
        )

    loss_fn = nn.BCELoss()
    patience = args.early_stopping_patience
    best_es_metric = float("inf")
    patience_counter = 0
    best_save_metric = float("inf")
    best_mvtec_auc = -1.0

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss_sum = 0.0
        num_batches = 0

        # DataLoader returns (prompt, query, mask); mask is aligned to query.
        for prompt, query, mask in tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            ncols=100,
            leave=False,
            disable=global_rank != 0,
        ):
            optimizer.zero_grad()
            prompt = prompt.to(device)
            query = query.to(device)
            mask = mask.to(device)
            mask_predict = model(query, prompt)
            loss = loss_fn(mask_predict, mask.clamp(0.0, 1.0))
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item()
            num_batches += 1

        scheduler.step()
        avg_epoch_loss = all_reduce_mean_epoch_loss(epoch_loss_sum, num_batches, device)

        vloss: float | None = None
        if val_loader is not None:
            val_sampler.set_epoch(epoch)
            vloss = eval_meta_model_avg_loss(model, val_loader, device, loss_fn, world_size)

        if global_rank == 0:
            metric_tag = "val_loss" if vloss is not None else "train_loss"
            metric_save = float(vloss) if vloss is not None else avg_epoch_loss
            if metric_save < best_save_metric:
                best_save_metric = metric_save
                torch.save(model.module.state_dict(), args.best_path)
                print(f"[best] {metric_tag}={metric_save:.6f} -> saved {args.best_path}", flush=True)

        # ========== Optional: MVTec eval each epoch (multi-rank avg) ==========
        if args.mvtec_eval_each_epoch:
            # Each rank runs with a different seed, then all_reduce for average; only rank0 saves best
            if args.mvtec_device.strip().lower() == "local":
                mv_device = device
            else:
                mv_device = torch.device(args.mvtec_device)

            # If mvtec_device differs from training device, copy weights per rank (slower)
            if mv_device != device:
                mv_model = MetaUAS("efficientnet-b4", "unet", 5, 5, 3, "sa", "cat").to(mv_device)
                mv_model.load_state_dict(model.module.state_dict(), strict=True)
            else:
                mv_model = model.module

            one_json = args.mvtec_oneprompt_json.strip() or None
            proot = args.mvtec_prompt_root.strip() or None
            if proot is None:
                proot = args.mvtec_root

            seed_this_rank = int(args.mvtec_seed) + int(global_rank)

            # Assign different oneprompt_seed*.json per rank (for multi-prompt averaging)
            seeds_csv = args.mvtec_oneprompt_seeds.strip()
            if seeds_csv:
                if one_json is None and not args.mvtec_oneprompt_dir.strip():
                    raise ValueError(
                        "--mvtec-oneprompt-seeds requires --mvtec-oneprompt-json or --mvtec-oneprompt-dir"
                    )
                seeds_list = [int(x.strip()) for x in seeds_csv.split(",") if x.strip()]
                if len(seeds_list) < world_size:
                    raise ValueError(
                        f"--mvtec-oneprompt-seeds needs at least {world_size} seeds (got: {len(seeds_list)})"
                    )
                one_dir = args.mvtec_oneprompt_dir.strip()
                if not one_dir:
                    one_dir = str(Path(one_json).expanduser().resolve().parent)
                chosen_seed = seeds_list[int(global_rank)]
                one_json = str(Path(one_dir) / f"oneprompt_seed{chosen_seed}.json")
            _, mean_auc_local = evaluate_mvtec(
                mv_model,
                mv_device,
                dataset_path=args.mvtec_root,
                normal_image_path=None,
                batch_size=args.mvtec_batch_size,
                seed=seed_this_rank,
                oneprompt_json=one_json,
                prompt_image_root=proot,
            )

            # Average across all ranks
            t = torch.tensor([float(mean_auc_local)], device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            mean_auc_avg = (t[0] / float(world_size)).item()

            if global_rank == 0:
                print(
                    f"[mvtec] epoch={epoch + 1} mean_auc_avg={mean_auc_avg:.4f} "
                    f"(seed_base={args.mvtec_seed}, world_size={world_size})",
                    flush=True,
                )
                if mean_auc_avg > best_mvtec_auc:
                    best_mvtec_auc = float(mean_auc_avg)
                    torch.save(model.module.state_dict(), args.best_mvtec_path)
                    print(
                        f"[best_mvtec] mean_auc_avg={mean_auc_avg:.4f} -> saved {args.best_mvtec_path}",
                        flush=True,
                    )

            if mv_device != device:
                del mv_model

        if patience > 0:
            ref_metric = float(vloss) if vloss is not None else avg_epoch_loss
            if ref_metric < best_es_metric:
                best_es_metric = ref_metric
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if global_rank == 0:
                        print(f"Early stopping at epoch {epoch + 1} (monitor={'val' if vloss is not None else 'train'})")
                    break

        if global_rank == 0:
            if not args.no_epoch_checkpoints:
                torch.save(model.module.state_dict(), f"{args.checkpoint_prefix}_{epoch + 1}.pth")
            lr_str = ""
            try:
                lr_str = f", LR: {scheduler.get_last_lr()[0]:.6f}"
            except Exception:
                pass
            msg = f"Epoch {epoch + 1}, train_loss: {avg_epoch_loss:.6f}{lr_str}"
            if vloss is not None:
                msg += f", val_loss: {vloss:.6f}"
            print(msg, flush=True)

    ddp_teardown()

    del model
    del dataloader
    del dataset_split
    del train_dataset


if __name__ == "__main__":
    main()
