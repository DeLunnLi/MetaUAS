"""
CYWS coco-inpainted dataset with MetaUAS paper-style synthesis.
- local-region: DRAEM/Perlin + DTD texture synthesis
- object-level: CYWS inpaint pairs or COCO instance paste exchange
- Returns (prompt, query, mask) aligned for model(query, prompt)
"""

from __future__ import annotations

import os
import pickle
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import numpy as np
import cv2
import imgaug.augmenters as iaa
from einops import rearrange
from PIL import Image, PngImagePlugin
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from .aug import UnifiedPipeline, rand_augmenter
from .perlin import rand_perlin_2d_np

import utils.general
import utils.geometry

PngImagePlugin.MAX_TEXT_CHUNK = 2 * (1024**2)

_PAIR_RE = re.compile(r"^(\d+)_mask(\d+)\.png$", re.IGNORECASE)
_STEM_RE = re.compile(r"^(\d+)_mask(\d+)$")


def _load_split_indices(root: Path, split: str) -> Set[int]:
    pkl = root / "data_split.pkl"
    if not pkl.is_file():
        raise FileNotFoundError(f"Missing data_split.pkl: {pkl}")
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    if split not in d:
        raise KeyError(f"data_split.pkl has no key {split!r}, only {list(d.keys())}")
    arr = d[split]
    return {int(x) for x in arr}


def group_inpainted_stems_by_idx(stems: List[str]) -> Dict[int, List[str]]:
    """Group all valid stems (``{idx}_mask{k}``) under the same COCO idx for inpaint-to-inpaint pairing."""
    by: Dict[int, List[str]] = {}
    for s in stems:
        m = _STEM_RE.match(s)
        if not m:
            continue
        idx = int(m.group(1))
        by.setdefault(idx, []).append(s)
    for k in list(by.keys()):
        by[k].sort()
    return by


def build_inpainted_pair_stems(root: Path, index_set: Set[int]) -> List[str]:
    """Scan inpainted/*.png, keep stems matching {idx}_mask{id} with idx in index_set where both images_and_masks/{idx}.png and images_and_masks/{stem}.png exist."""
    inp_dir = root / "inpainted"
    img_dir = root / "images_and_masks"
    if not inp_dir.is_dir():
        raise FileNotFoundError(f"Missing inpainted directory: {inp_dir}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing images_and_masks directory: {img_dir}")

    stems: List[str] = []
    for name in os.listdir(inp_dir):
        if not name.lower().endswith(".png"):
            continue
        m = _PAIR_RE.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx not in index_set:
            continue
        stem = Path(name).stem
        if not (img_dir / f"{idx}.png").is_file():
            continue
        if not (img_dir / f"{stem}.png").is_file():
            continue
        stems.append(stem)

    stems.sort()
    if not stems:
        raise ValueError(f"No valid inpainted pairs found under {root} (check data_split and file integrity)")
    return stems


def _scan_coco_images_for_exchange(coco_images_dir: Path) -> List[Tuple[str, str]]:
    """Return [(absolute_path, stem), ...] for exchange random source images."""
    if not coco_images_dir.is_dir():
        return []
    out: List[Tuple[str, str]] = []
    for pat in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
        for p in sorted(coco_images_dir.glob(pat)):
            out.append((str(p.resolve()), p.stem))
    return out


def paste_random_coco_instances(
    base_image_chw: torch.Tensor,
    coco_files: List[Tuple[str, str]],
    metadata_root: Path,
    exchange_max_area_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Equivalent to CocoMetaDataset.get_objects_added_image: paste random instance masks from another COCO image onto base_image_chw. Returns (base, zero mask) on failure."""
    hw = base_image_chw.shape[-2:]
    h, w = int(hw[0]), int(hw[1])
    max_pixels = float(h * w) * float(exchange_max_area_ratio)
    if not coco_files:
        return base_image_chw, torch.zeros((h, w), dtype=torch.bool, device=base_image_chw.device)

    for _ in range(48):
        path, stem = random.choice(coco_files)
        src = Image.open(path).convert("RGB")
        select_image = pil_to_tensor(src).float() / 255.0

        ann_path = metadata_root / f"{stem}.npy"
        if not ann_path.is_file():
            continue
        annotations = np.load(ann_path, allow_pickle=True)
        if len(annotations) == 0:
            continue

        order = list(range(len(annotations)))
        random.shuffle(order)
        selected_indices: list[int] = []
        for i in order:
            trial = selected_indices + [i]
            trial_anns = [annotations[j] for j in trial]
            _, ann_resized = utils.geometry.resize_image_and_annotations(select_image, (h, w), trial_anns)
            trial_mask = utils.general.coco_annotations_to_mask_np_array(ann_resized, (h, w))
            if float(trial_mask.sum()) <= max_pixels:
                selected_indices.append(i)

        if not selected_indices:
            continue

        selected_annotations = [annotations[j] for j in selected_indices]
        select_image_resized_to_current, annotations_resized = utils.geometry.resize_image_and_annotations(
            select_image, (h, w), selected_annotations
        )

        annotation_mask = utils.general.coco_annotations_to_mask_np_array(annotations_resized, (h, w))
        annotation_mask = torch.from_numpy(annotation_mask).to(torch.bool)
        if int(annotation_mask.sum()) == 0:
            continue

        image_tensor_copy = base_image_chw.clone()
        image_tensor_copy = rearrange(image_tensor_copy, "c h w -> h w c")
        select_image_resized_to_current = rearrange(select_image_resized_to_current, "c h w -> h w c")
        image_tensor_copy[annotation_mask] = select_image_resized_to_current[annotation_mask]

        return rearrange(image_tensor_copy, "h w c -> c h w"), annotation_mask

    return base_image_chw, torch.zeros((h, w), dtype=torch.bool, device=base_image_chw.device)


class CocoInpaintedPairDataset(torch.utils.data.Dataset):
    """Returns (prompt, query, mask) — mask is always in query pixel coordinates, aligned to MetaUAS.forward(query, prompt)."""

    def __init__(
        self,
        pair_stems: List[str],
        root: Union[str, Path],
        target_size: Union[int, Tuple[int, int]],
        aug_kwargs: Optional[Dict[str, Any]] = None,
        *,
        local_region_p: float = 0.5,
        dtd_root: str = "/path/to/dtd/images",
        exchange_p: float = 0.5,
        coco_exchange_images_dir: str = "",
        coco_exchange_metadata_dir: str = "",
        exchange_max_area_ratio: float = 0.35,
    ):
        self.pair_stems = list(pair_stems)
        self.root = Path(root)
        self.local_region_p = float(local_region_p)
        self.exchange_p = float(exchange_p)
        self.exchange_max_area_ratio = float(exchange_max_area_ratio)
        self._stems_by_idx: Dict[int, List[str]] = group_inpainted_stems_by_idx(self.pair_stems)
        self.dtd_root = str(dtd_root)
        # local-region change (DRAEM-style) uses DTD textures
        self._dtd_paths = sorted(str(p) for p in Path(self.dtd_root).glob("*/*.jpg"))
        if not self._dtd_paths:
            raise FileNotFoundError(
                f"No DTD textures found under {self.dtd_root} (expected */*.jpg). "
                "Set local_region_p=0 to disable the local-region branch."
            )
        self.aug = UnifiedPipeline(
            mode="registered",
            target_size=target_size,
            **(aug_kwargs or {}),
        )
        if isinstance(target_size, (tuple, list)):
            self._ts = (int(target_size[0]), int(target_size[1]))
        else:
            self._ts = (int(target_size), int(target_size))
        self._resize = transforms.Resize(self._ts)
        self._mask_resize = transforms.Resize(self._ts, interpolation=InterpolationMode.NEAREST)
        self._rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

        self._coco_files: List[Tuple[str, str]] = []
        self._metadata_root = Path(coco_exchange_metadata_dir).expanduser() if coco_exchange_metadata_dir else Path()
        img_dir = Path(coco_exchange_images_dir).expanduser() if coco_exchange_images_dir else Path()
        if self.exchange_p > 0.0:
            self._coco_files = _scan_coco_images_for_exchange(img_dir)
            if not self._coco_files or not self._metadata_root.is_dir():
                self.exchange_p = 0.0

    def __len__(self) -> int:
        return len(self.pair_stems)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        im = Image.open(path).convert("RGB")
        return pil_to_tensor(im).float() / 255.0

    def _load_mask_long(self, path: Path) -> torch.Tensor:
        im = Image.open(path).convert("L")
        t = pil_to_tensor(im).long()
        return (t > 0).long()

    def _cyws_random_two_views(self, idx: int, stem: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniformly sample two distinct views (original + inpainted stems) returning (img_a, img_b, mask_long)."""
        stems_list = sorted(self._stems_by_idx.get(idx, []) or [])
        if not stems_list:
            stems_list = [stem]
        elif stem not in stems_list:
            stems_list = sorted(set(stems_list) | {stem})

        view_ids = [-1] + list(range(len(stems_list)))
        a, b = random.sample(view_ids, 2)

        def rgb_view(vid: int) -> torch.Tensor:
            if vid == -1:
                return self._load_rgb(self.root / "images_and_masks" / f"{idx}.png")
            return self._load_rgb(self.root / "inpainted" / f"{stems_list[vid]}.png")

        def mask_view(vid: int) -> Optional[torch.Tensor]:
            if vid == -1:
                return None
            m = self._load_mask_long(self.root / "images_and_masks" / f"{stems_list[vid]}.png")
            return m[0] if m.dim() == 3 else m

        img_a = rgb_view(a)
        img_b = rgb_view(b)
        if img_a.shape != img_b.shape:
            st0 = stems_list[0]
            o = self._load_rgb(self.root / "images_and_masks" / f"{idx}.png")
            i0 = self._load_rgb(self.root / "inpainted" / f"{st0}.png")
            mk = self._load_mask_long(self.root / "images_and_masks" / f"{st0}.png")
            mk = mk[0] if mk.dim() == 3 else mk
            return o, i0, mk

        if a == -1 or b == -1:
            v_inp = b if a == -1 else a
            mk = mask_view(v_inp)
            assert mk is not None
            mask_long = mk.to(torch.long)
        else:
            ma = mask_view(a)
            mb = mask_view(b)
            assert ma is not None and mb is not None
            if ma.shape != mb.shape:
                st0 = stems_list[0]
                o = self._load_rgb(self.root / "images_and_masks" / f"{idx}.png")
                i0 = self._load_rgb(self.root / "inpainted" / f"{st0}.png")
                mk = self._load_mask_long(self.root / "images_and_masks" / f"{st0}.png")
                mk = mk[0] if mk.dim() == 3 else mk
                return o, i0, mk
            mask_long = ((ma > 0) ^ (mb > 0)).to(torch.long)

        return img_a, img_b, mask_long

    def _synthesize_local_region_change(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Local-Region Change (DRAEM-style): synthesize anomaly in 256x256 center-crop space, then resize back to original resolution."""
        _, h0, w0 = image_tensor.shape
        pil_full = to_pil_image(image_tensor.clamp(0.0, 1.0))
        pil_small = transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR)(pil_full)
        pil_small = transforms.CenterCrop(256)(pil_small)
        image_tensor = pil_to_tensor(pil_small).float() / 255.0

        # numpy HWC, synthesize in 256 space
        c, h, w = int(image_tensor.shape[0]), int(image_tensor.shape[1]), int(image_tensor.shape[2])
        image = image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.float32)  # HWC, 0-1

        # Load and augment texture source
        anomaly_source_path = random.choice(self._dtd_paths)
        aug = rand_augmenter()
        anomaly_source_img = cv2.imread(anomaly_source_path)
        if anomaly_source_img is None:
            raise RuntimeError(f"Failed to read DTD texture: {anomaly_source_path}")
        anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_BGR2RGB)
        anomaly_source_img = cv2.resize(anomaly_source_img, (w, h))
        anomaly_img_augmented = aug(image=anomaly_source_img).astype(np.float32) / 255.0  # HWC

        # Perlin mask
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** int(torch.randint(min_perlin_scale, perlin_scale, (1,)).item())
        perlin_scaley = 2 ** int(torch.randint(min_perlin_scale, perlin_scale, (1,)).item())
        perlin_noise = rand_perlin_2d_np((h, w), (perlin_scalex, perlin_scaley))
        perlin_noise = self._rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = (perlin_noise > threshold).astype(np.float32)  # HW

        beta = float(torch.rand(1).item() * 0.8)
        mask3 = np.repeat(perlin_thr[..., None], repeats=c, axis=2)
        changed = (image * (1.0 - mask3) + (1.0 - beta) * mask3 * anomaly_img_augmented + beta * image * mask3).astype(
            np.float32
        )

        # Resize back to original resolution (cv2: width, height)
        if h0 != h or w0 != w:
            changed_full = cv2.resize(changed, (int(w0), int(h0)), interpolation=cv2.INTER_LINEAR)
            mk_full = cv2.resize(perlin_thr, (int(w0), int(h0)), interpolation=cv2.INTER_LINEAR)
            perlin_thr = (mk_full > 0.5).astype(np.float32)
            changed = changed_full.astype(np.float32)

        changed_t = torch.from_numpy(changed).permute(2, 0, 1).contiguous().float()
        mask_t = torch.from_numpy(perlin_thr).contiguous().float()
        return changed_t, mask_t

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.pair_stems[index]
        sm = _STEM_RE.match(stem)
        if not sm:
            raise ValueError(f"Invalid stem: {stem}")
        idx = int(sm.group(1))

        orig_path = self.root / "images_and_masks" / f"{idx}.png"
        image = self._load_rgb(orig_path)
        # Training: choose object-level / local-region with ~50% probability
        do_local = (self.local_region_p > 0.0) and (random.random() < self.local_region_p)
        if do_local:
            changed, lr_mask = self._synthesize_local_region_change(image)
            image, changed, mask_on_image, mask_on_changed, _ = self.aug.forward_local_region(
                image, changed, lr_mask
            )
            # out0=prompt, out1=query; mask aligned to query
            if random.random() < 0.5:
                out0 = self._resize(image)
                out1 = self._resize(changed)
                query_m = mask_on_changed
            else:
                out0 = self._resize(changed)
                out1 = self._resize(image)
                query_m = mask_on_image
            out_m = self._mask_resize(query_m.float().unsqueeze(0)).squeeze(0)
        else:
            # object-level: exchange (COCO copy-paste) vs CYWS inpaint pairs
            do_exchange = (
                self.exchange_p > 0.0
                and len(self._coco_files) > 0
                and self._metadata_root.is_dir()
                and (random.random() < self.exchange_p)
            )
            if do_exchange:
                src_image, src_mask = paste_random_coco_instances(
                    image,
                    self._coco_files,
                    self._metadata_root,
                    self.exchange_max_area_ratio,
                )
                if int(src_mask.sum()) == 0:
                    do_exchange = False
            if do_exchange:
                exchanged, base, mask_ex, mask_base, _ = self.aug.forward_object_exchange(
                    image, src_image, src_mask.float()
                )
                # query = out1; mask_ex aligned to exchanged, mask_base aligned to base
                if random.random() < 0.5:
                    out0 = self._resize(exchanged)
                    out1 = self._resize(base)
                    query_m = mask_base
                else:
                    out0 = self._resize(base)
                    out1 = self._resize(exchanged)
                    query_m = mask_ex
                out_m = self._mask_resize(query_m.float().unsqueeze(0)).squeeze(0)
            else:
                img1, img2, mask_long = self._cyws_random_two_views(idx, stem)
                if mask_long.dim() == 3:
                    mask_long = mask_long[0]
                image, inpainted, image_mask, inpainted_image_mask, _ = self.aug.forward_object_level(
                    img1, img2, mask_long
                )

                # image/image_mask aligned to img1, inpainted/other-mask aligned to img2
                # apply_geometric_sync intersected masks in original coords then warped; pick the one aligned to query (out1)
                swap = random.random() < 0.5
                if swap:
                    out0 = self._resize(image)
                    out1 = self._resize(inpainted)
                    query_m = inpainted_image_mask
                else:
                    out0 = self._resize(inpainted)
                    out1 = self._resize(image)
                    query_m = image_mask
                out_m = self._mask_resize(query_m.float().unsqueeze(0)).squeeze(0)

        return out0, out1, out_m


def make_coco_inpainted_train_loader(
    root: str,
    target_size: Union[int, Tuple[int, int]],
    batch_size: int,
    world_size: int,
    local_rank: int,
    num_workers: int = 8,
    pin_memory: bool = True,
    aug_kwargs: Optional[Dict[str, Any]] = None,
    local_region_p: float = 0.5,
    dtd_root: str = "/path/to/dtd/images",
    exchange_p: float = 0.5,
    coco_exchange_images_dir: str = "",
    coco_exchange_metadata_dir: str = "",
    exchange_max_area_ratio: float = 0.35,
) -> Tuple[Dict[str, Any], CocoInpaintedPairDataset, DistributedSampler, DataLoader]:
    root_p = Path(root).expanduser().resolve()
    train_idx = _load_split_indices(root_p, "train")
    stems = build_inpainted_pair_stems(root_p, train_idx)
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    ds = CocoInpaintedPairDataset(
        stems,
        root_p,
        target_size,
        aug_kwargs=aug_kwargs,
        local_region_p=float(local_region_p),
        dtd_root=str(dtd_root),
        exchange_p=float(exchange_p),
        coco_exchange_images_dir=str(coco_exchange_images_dir),
        coco_exchange_metadata_dir=str(coco_exchange_metadata_dir),
        exchange_max_area_ratio=float(exchange_max_area_ratio),
    )
    sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
    )
    loader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    meta = {
        "root": str(root_p),
        "num_pairs": len(stems),
        "split": "train",
        "local_region_p": float(local_region_p),
        "dtd_root": str(dtd_root),
        "exchange_p": float(exchange_p),
        "coco_exchange_images_dir": str(coco_exchange_images_dir),
        "coco_exchange_metadata_dir": str(coco_exchange_metadata_dir),
        "exchange_max_area_ratio": float(exchange_max_area_ratio),
    }
    return meta, ds, sampler, loader


def make_coco_inpainted_val_loader(
    root: str,
    target_size: Union[int, Tuple[int, int]],
    batch_size: int,
    world_size: int,
    local_rank: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    aug_kwargs: Optional[Dict[str, Any]] = None,
    local_region_p: float = 0.5,
    dtd_root: str = "/path/to/dtd/images",
    exchange_p: float = 0.5,
    coco_exchange_images_dir: str = "",
    coco_exchange_metadata_dir: str = "",
    exchange_max_area_ratio: float = 0.35,
) -> Tuple[CocoInpaintedPairDataset, DistributedSampler, DataLoader]:
    root_p = Path(root).expanduser().resolve()
    val_idx = _load_split_indices(root_p, "val")
    stems = build_inpainted_pair_stems(root_p, val_idx)
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    ds = CocoInpaintedPairDataset(
        stems,
        root_p,
        target_size,
        aug_kwargs=aug_kwargs,
        local_region_p=float(local_region_p),
        dtd_root=str(dtd_root),
        exchange_p=float(exchange_p),
        coco_exchange_images_dir=str(coco_exchange_images_dir),
        coco_exchange_metadata_dir=str(coco_exchange_metadata_dir),
        exchange_max_area_ratio=float(exchange_max_area_ratio),
    )
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=local_rank, shuffle=False)
    loader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return ds, sampler, loader
