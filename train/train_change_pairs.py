#!/usr/bin/env python3
"""
MetaUAS DDP 训练：CYWS coco-inpainted 管线（data_split.pkl 划分 train/val）。

变化分割设定，数据源为 coco-inpainted 目录结构。
- object-level：按 exchange_p 在 COCO 实例粘贴 exchange 与 CYWS inpaint 成对间划分
- CYWS inpaint：随机两视图，mask 异或得变化区后做统一几何增强
- DataLoader 返回 (prompt, query, mask)，mask 与 query 对齐

启动：torchrun --nproc_per_node=4 train/train_change_pairs.py --coco-inpainted-root /path/to/coco-inpainted/train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保可 import 顶层包：dataset/ module/ utils/
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
        description="MetaUAS DDP：CYWS coco-inpainted 数据（参见 arXiv:2505.09265）"
    )
    p.add_argument("--target-size", type=int, default=256)
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="每卡 batch（DDP）；B4+双图+cat 在 24GB 上 128 易 OOM，可再降到 16",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="默认 30 与论文 (arXiv:2505.09265) 训练 epoch 数一致",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        dest="weight_decay",
        help="AdamW weight decay；论文为 0.0005",
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--scheduler",
        type=str,
        choices=("exponential", "cosine", "constant"),
        default="cosine",
        help="constant：全程固定学习率（每 epoch 调用 step 但 lr 不变）",
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
        help="每 epoch 后做验证（关闭: --no-val-each-epoch）。验证需各 rank 都参与 all_reduce。",
    )
    p.add_argument(
        "--val-batch-size",
        type=int,
        default=32,
        help="验证每卡 batch；可与训练 batch 分开以省显存",
    )

    # ========= MVTec eval each epoch (rank0 only) =========
    p.add_argument(
        "--mvtec-eval-each-epoch",
        action="store_true",
        help="每个 epoch 后在 MVTec 上评估（rank0），并按 mean_auc 保存最优 ckpt",
    )
    p.add_argument(
        "--mvtec-root",
        type=str,
        default=str(Path.home() / "datasets/mvtec_ad"),
        help="MVTec AD 根目录（含 bottle/ cable/ ...）",
    )
    p.add_argument(
        "--mvtec-device",
        type=str,
        default="local",
        help="MVTec 评估所用 device：'local' 表示每个 rank 用本地训练卡（推荐）；或显式如 cuda:0",
    )
    p.add_argument("--mvtec-batch-size", type=int, default=32)
    p.add_argument(
        "--mvtec-seed",
        type=int,
        default=1,
        help="MVTec 评估 seed 基准值；若启用多卡平均，则各 rank 用 seed+global_rank",
    )
    p.add_argument(
        "--mvtec-oneprompt-json",
        type=str,
        default="",
        help="若提供：使用 oneprompt_seed*.json 为每类固定 prompt（优先于 seed 抽样）",
    )
    p.add_argument(
        "--mvtec-oneprompt-seeds",
        type=str,
        default="",
        help=(
            "多卡评估时为各 rank 指定不同 oneprompt seed（逗号分隔，如 '1,2,3,5'）。"
            "若设置且提供 --mvtec-oneprompt-json/--mvtec-oneprompt-dir，则 rank i 使用 seeds[i]。"
        ),
    )
    p.add_argument(
        "--mvtec-oneprompt-dir",
        type=str,
        default="",
        help="oneprompt_seed*.json 所在目录（可选；为空则从 --mvtec-oneprompt-json 推断目录）",
    )
    p.add_argument(
        "--mvtec-prompt-root",
        type=str,
        default="",
        help="oneprompt json 中 filename 的根目录；空则默认等于 --mvtec-root",
    )
    p.add_argument(
        "--best-mvtec-path",
        type=str,
        default="best_model_mvtec_auc.pth",
        help="按 MVTec mean_auc 保存的最优 ckpt（rank0 写入）",
    )

    # ========= Augmentations (CYWS-style, controlled via CLI) =========
    p.add_argument("--aug-crop-p", type=float, default=0.0)
    p.add_argument(
        "--aug-crop-independent",
        action="store_true",
        help="若开启：两张图独立采样 crop，并用 CYWS 交集逻辑严格对齐 mask（更难）",
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
        help="数据集根目录（含 data_split.pkl、images_and_masks、inpainted）",
    )
    p.add_argument(
        "--coco-inpainted-local-region-p",
        type=float,
        default=0.5,
        help="以该概率走 local-region(DRAEM/Perlin) 合成；其余为 object-level(CYWS inpaint / exchange)",
    )
    p.add_argument(
        "--coco-inpainted-dtd-root",
        type=str,
        default="/home/ldl/datasets/dtd/images",
        help="local-region 分支纹理源（DTD）根目录",
    )
    p.add_argument(
        "--coco-inpainted-exchange-p",
        type=float,
        default=0.5,
        help="object-level 内：以该概率走 COCO 实例粘贴 exchange；否则走 CYWS inpaint（需有效 COCO 图+meta）",
    )
    p.add_argument(
        "--coco-inpainted-coco-images",
        type=str,
        default="/home/ldl/datasets/images/train2017",
        help="exchange 源：MS-COCO train 图像目录（*.jpg 等）",
    )
    p.add_argument(
        "--coco-inpainted-coco-metadata",
        type=str,
        default="/home/ldl/datasets/images/meta_data",
        help="exchange 源：与 pre_process_coco 一致的 *.npy 实例标注目录",
    )
    p.add_argument(
        "--coco-inpainted-exchange-max-area",
        type=float,
        default=0.35,
        help="exchange 粘贴区域面积上限（相对整图）",
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
    # UnetDecoder 在 skip=None 时会跳过 SCSE 的 attention1，部分参数可能本轮无 grad，需打开 unused 检测
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # MetaUAS 中 encoder 已冻结；仅优化 alignment / decoder / segmentation head（与论文一致）
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
        raise ValueError("需要非空的 --coco-inpainted-root")

    if float(args.coco_inpainted_exchange_p) > 0.0:
        ex_img = Path(str(args.coco_inpainted_coco_images).strip())
        ex_meta = Path(str(args.coco_inpainted_coco_metadata).strip())
        if (not ex_img.is_dir()) or (not ex_meta.is_dir()) or (not _scan_coco_images_for_exchange(ex_img)):
            if global_rank == 0:
                print(
                    "[train_change_pairs] exchange 已关闭（COCO 图像目录或 meta_data 无效，"
                    "或目录内无图像）。仅 local-region + CYWS inpaint。",
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
            f"[train_change_pairs] encoder 冻结 | AdamW 可训练参数 {n_tr:,} / 冻结 {n_fr:,} "
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

        # DataLoader：(prompt, query, mask)，mask 与 query 对齐（与各 Dataset 约定一致）。
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
            # 每个 rank 跑不同 seed，然后 all_reduce 求平均；只由 rank0 保存 best
            if args.mvtec_device.strip().lower() == "local":
                mv_device = device
            else:
                mv_device = torch.device(args.mvtec_device)

            # 若 mvtec_device 与训练 device 不同，每个 rank 都需要临时搬运权重（更慢）
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

            # 做法A：每个 rank 绑定不同 oneprompt_seed*.json（用于多 prompt 平均）
            seeds_csv = args.mvtec_oneprompt_seeds.strip()
            if seeds_csv:
                if one_json is None and not args.mvtec_oneprompt_dir.strip():
                    raise ValueError(
                        "--mvtec-oneprompt-seeds 需要同时提供 --mvtec-oneprompt-json 或 --mvtec-oneprompt-dir"
                    )
                seeds_list = [int(x.strip()) for x in seeds_csv.split(",") if x.strip()]
                if len(seeds_list) < world_size:
                    raise ValueError(
                        f"--mvtec-oneprompt-seeds 至少需要 {world_size} 个 seed（当前: {len(seeds_list)}）"
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

            # 求多卡平均
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
