from __future__ import annotations

import glob
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageFilter, PngImagePlugin
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor

import utils.general
import utils.geometry
from .aug import UnifiedPipeline, rand_augmenter
from .perlin import rand_perlin_2d_np


@dataclass(frozen=True)
class CocoMetaDataPaths:
    """COCO synthesis/inpainting training data root paths (``path_to_dtd`` is DTD textures, not COCO)."""

    coco_path_to_images: str
    coco_path_to_masks: str
    coco_path_to_inpainted: str
    path_to_dtd: str
    coco_path_to_metadata: str
    coco_change_pairs_root: Optional[str] = None


DEFAULT_COCO_META_PATHS = CocoMetaDataPaths(
    coco_path_to_images="/home/ldl/datasets/images/train2017",
    coco_path_to_masks="/home/ldl/datasets/images/mask2",
    coco_path_to_inpainted="/home/ldl/datasets/images/inpainted_images_mask2",
    path_to_dtd="/home/ldl/datasets/dtd/images",
    coco_path_to_metadata="/home/ldl/datasets/images/meta_data",
    coco_change_pairs_root=None,
)

coco_path_to_images = DEFAULT_COCO_META_PATHS.coco_path_to_images
coco_path_to_masks = DEFAULT_COCO_META_PATHS.coco_path_to_masks
coco_path_to_inpainted = DEFAULT_COCO_META_PATHS.coco_path_to_inpainted
path_to_dtd = DEFAULT_COCO_META_PATHS.path_to_dtd
coco_path_to_metadata = DEFAULT_COCO_META_PATHS.coco_path_to_metadata
PngImagePlugin.MAX_TEXT_CHUNK = 2 * (1024**2)


# return four task image pair and mask
class CocoMetaDataset(Dataset):
    def __init__(
        self,
        image_name_list,
        target_size,
        paths: Optional[CocoMetaDataPaths] = None,
        image_stem_allowlist: Optional[Set[str]] = None,
    ):
        self.paths = paths if paths is not None else DEFAULT_COCO_META_PATHS
        names = list(image_name_list)
        if image_stem_allowlist is not None:
            names = [n for n in names if Path(n).stem in image_stem_allowlist]
        if len(names) == 0:
            raise ValueError(
                "CocoMetaDataset: zero samples after filtering. Check split file and --stem-allowlist."
            )
        self.image_names = names
        self.images_len = len(self.image_names)
        self.index_to_image_name = {idx: img_name for idx, img_name in enumerate(names)}
        self.anomaly_source_paths = sorted(glob.glob(self.paths.path_to_dtd + "/*/*.jpg"))
        self.image_augmentations = UnifiedPipeline(mode="registered")
        self.target_size = target_size
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return self.images_len

    def __getitem__(self, index):
        return self.__base_getitem__(index)

    def __base_getitem__(self, index):
        image = self.get_image_as_tensor(index)
        resize = transforms.Resize(self.target_size)
        mask_resized = transforms.Resize(self.target_size, Image.NEAREST)
        if random.random() < 0.4:
            dtd_image, dtd_mask = self.get_berlin_dtd_added_image(index)
            image_transformed, dtd_image_transformed, _, dtd_mask_transformed, transform1 = self.image_augmentations.forward_transformed(
                image, dtd_image, dtd_mask
            )
            image_transformed = image_transformed.squeeze(0)
            dtd_image_transformed = dtd_image_transformed.squeeze(0)
            dtd_mask_transformed = dtd_mask_transformed.squeeze(0)
            return resize(image_transformed), resize(dtd_image_transformed), mask_resized(dtd_mask_transformed)
        elif random.random() < 0.5:
            exchanged_image, exchanged_mask = self.get_objects_added_image(image, index)
            image_transformed, exchanged_image_transformed, _, exchanged_mask_transformed, transform1 = self.image_augmentations.forward_transformed(
                image, exchanged_image, exchanged_mask
            )
            image_transformed = image_transformed.squeeze(0)
            exchanged_image_transformed = exchanged_image_transformed.squeeze(0)
            exchanged_mask_transformed = exchanged_mask_transformed.squeeze(0)
            return resize(image_transformed), resize(exchanged_image_transformed), mask_resized(exchanged_mask_transformed)
        else:
            inpainted_image, inpainted_mask = self.get_inpainted_image_mask_as_tensor(index)
            image, inpainted_image, image_mask, inpainted_image_mask, transform1 = self.image_augmentations.forward_transformed(
                image, inpainted_image, inpainted_mask
            )
            image = image.squeeze(0)
            inpainted_image = inpainted_image.squeeze(0)
            image_mask = image_mask.squeeze(0)
            inpainted_image_mask = inpainted_image_mask.squeeze(0)
            if random.random() < 0.5:
                return resize(image), resize(inpainted_image), mask_resized(inpainted_image_mask)
            else:
                return resize(inpainted_image), resize(image), mask_resized(image_mask)

    def get_image_as_tensor(self, index):
        image_name = self.index_to_image_name[index]
        path_to_image = os.path.join(self.paths.coco_path_to_images, image_name)
        image = Image.open(path_to_image).convert("RGB")
        image_tensor = pil_to_tensor(image).float() / 255.0
        return image_tensor

    def _try_load_inpainted_from_change_pairs(self, stem: str):
        root = self.paths.coco_change_pairs_root
        if not root:
            return None
        root_p = Path(root).expanduser()
        if not root_p.is_dir():
            return None
        b_path = None
        l_path = None
        for fe in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            bp = root_p / "B" / f"{stem}{fe}"
            if bp.is_file():
                b_path = bp
                break
        for fe in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            lp = root_p / "label" / f"{stem}{fe}"
            if lp.is_file():
                l_path = lp
                break
        if b_path is None or l_path is None:
            return None
        inpainted_image = Image.open(b_path).convert("RGB")
        image_tensor = pil_to_tensor(inpainted_image).float() / 255.0
        pil_mask = Image.open(l_path).convert("L")
        pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=2))
        mask_as_tensor = pil_to_tensor(pil_mask).long()
        mask_as_tensor = (mask_as_tensor > 0).long()
        return image_tensor, mask_as_tensor

    def get_inpainted_image_mask_as_tensor(self, index):
        image_name = self.index_to_image_name[index]
        stem = Path(image_name).stem

        cp = self._try_load_inpainted_from_change_pairs(stem)
        if cp is not None:
            return cp

        base_name, ext = image_name.split(".")
        inpainted_image_name = f"{base_name}_mask2.{ext}"

        path_to_inpainted_image = os.path.join(self.paths.coco_path_to_inpainted, inpainted_image_name)
        path_to_inpainted_mask = os.path.join(self.paths.coco_path_to_masks, inpainted_image_name)

        inpainted_image = Image.open(path_to_inpainted_image).convert("RGB")
        image_tensor = pil_to_tensor(inpainted_image).float() / 255.0

        pil_mask = Image.open(path_to_inpainted_mask).convert("L")  # Convert to grayscale ('L' mode)
        pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=2))
        mask_as_tensor = pil_to_tensor(pil_mask).long()
        mask_as_tensor = (mask_as_tensor > 0).long()

        return image_tensor, mask_as_tensor

    def get_objects_added_image(self, image_tensor, index):
        # select another image for objects exchange
        index_except_current = list(range(index)) + list(range(index + 1, self.images_len))
        random_image_index = random.choice(index_except_current)

        select_image_name = self.image_names[random_image_index]
        select_image = self.get_image_as_tensor(random_image_index)

        annotation_name = os.path.splitext(select_image_name)[0] + ".npy"
        annotation_path = os.path.join(self.paths.coco_path_to_metadata, annotation_name)
        annotations = np.load(annotation_path, allow_pickle=True)

        if len(annotations) > 0:
            num_selected_instances = random.randint(1, len(annotations))
            selected_indices = random.sample(range(len(annotations)), num_selected_instances)

            # Get the selected annotations
            selected_annotations = [annotations[i] for i in selected_indices]

            # Resize the selected annotations and image
            select_image_resized_to_current, annotations_resized = utils.geometry.resize_image_and_annotations(
                select_image, image_tensor.shape[-2:], selected_annotations
            )

            annotation_mask = utils.general.coco_annotations_to_mask_np_array(annotations_resized, image_tensor.shape[-2:])
            annotation_mask = torch.from_numpy(annotation_mask).to(torch.bool)

            # Create a copy of the image tensor to modify
            image_tensor_copy = image_tensor.clone()
            image_tensor_copy = rearrange(image_tensor_copy, "c h w -> h w c")
            select_image_resized_to_current = rearrange(select_image_resized_to_current, "c h w -> h w c")

            # Apply the object exchange to the selected masks
            image_tensor_copy[annotation_mask] = select_image_resized_to_current[annotation_mask]

            return rearrange(image_tensor_copy, "h w c -> c h w"), annotation_mask
        else:
            # If no annotations are available, return the original image
            return image_tensor, torch.zeros_like(image_tensor[0], dtype=torch.bool)

    def get_berlin_dtd_added_image(self, index):
        image_name = self.index_to_image_name[index]
        path_to_image = os.path.join(self.paths.coco_path_to_images, image_name)

        transform_resize_crop = transforms.Compose(
            [
                transforms.Resize((256, 256), Image.LANCZOS),
                transforms.CenterCrop(256),
            ]
        )

        # Force RGB (grayscale-compatible)
        image = Image.open(path_to_image).convert("RGB")
        H_orig, W_orig = image.size
        image = transform_resize_crop(image)
        image = np.asarray(image).astype(np.float32) / 255.0
        num_channels = image.shape[-1]

        # Load and augment anomaly source image
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        anomaly_source_path = self.anomaly_source_paths[anomaly_source_idx]
        aug = rand_augmenter()

        anomaly_source_img = cv2.imread(anomaly_source_path)
        if num_channels == 1:
            anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_BGR2GRAY)
            anomaly_source_img = np.expand_dims(anomaly_source_img, axis=-1)
        else:
            anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_BGR2RGB)
        anomaly_source_img = cv2.resize(anomaly_source_img, (256, 256))
        anomaly_img_augmented = aug(image=anomaly_source_img).astype(np.float32) / 255.0

        # Perlin noise mask
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_noise = rand_perlin_2d_np((256, 256), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)

        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, 1.0, 0.0)
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        perlin_thr = np.repeat(perlin_thr, repeats=num_channels, axis=2)

        # Synthesize augmented image
        beta = torch.rand(1).item() * 0.8
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * perlin_thr * anomaly_img_augmented + beta * image * perlin_thr
        mask = (perlin_thr > 0.5).astype(np.float32)

        # Resize back to original dimensions
        augmented_image = cv2.resize(augmented_image, (H_orig, W_orig), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (H_orig, W_orig), interpolation=cv2.INTER_LINEAR)

        if num_channels == 1:
            augmented_image = np.expand_dims(augmented_image, axis=-1)
            mask = np.expand_dims(mask, axis=-1)

        augmented_image = augmented_image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1) if mask.ndim == 3 else mask

        mask_single_channel = mask[0, :, :] if mask.ndim == 3 else mask.squeeze()
        augmented_image = torch.from_numpy(augmented_image).float()
        mask_single_channel = torch.from_numpy(mask_single_channel).float()
        return augmented_image, mask_single_channel

def load_split(path_to_dataset, path_to_split=None, train_rate=0.10):
    """Load or create a train/val split, returning a dict with 'train' and 'val' keys."""

    if path_to_split:
        if os.path.exists(path_to_split):
            with open(path_to_split, "rb") as file:
                return pickle.load(file)

    images_list = [f for f in os.listdir(path_to_dataset)]
    list_len = len(images_list)
    train_size = int(list_len * train_rate)

    random.shuffle(images_list)

    train_images = images_list[:train_size]
    val_images = images_list[train_size:]

    split_filename = f"data_split_train_{train_rate}_val_{1-train_rate}.pkl"
    parent_folder = os.path.dirname(path_to_dataset)
    split_file_path = os.path.join(parent_folder, split_filename)

    train_val_split = {"train": train_images, "val": val_images}

    with open(split_file_path, "wb") as file:
        pickle.dump(train_val_split, file)

    return train_val_split