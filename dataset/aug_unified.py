from __future__ import annotations

import imgaug.augmenters as iaa
import kornia as K
import numpy as np
import torch
import torch.nn.functional as F


# DTD texture augmenters for local-region change
augmenters_dtd = [
    iaa.GammaContrast((0.5, 2.0), per_channel=True),
    iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
    iaa.pillike.EnhanceSharpness(),
    iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
    iaa.Solarize(0.5, threshold=(32, 128)),
    iaa.Posterize(),
    iaa.Invert(),
    iaa.pillike.Autocontrast(),
    iaa.pillike.Equalize(),
    iaa.Affine(rotate=(-45, 45)),
]


def rand_augmenter():
    """RandAugment: randomly select 3 augmenters for DTD-based image synthesis."""
    aug_ind = np.random.choice(np.arange(len(augmenters_dtd)), 3, replace=False)
    aug = iaa.Sequential([
        augmenters_dtd[aug_ind[0]],
        augmenters_dtd[aug_ind[1]],
        augmenters_dtd[aug_ind[2]],
    ])
    return aug


class UnifiedPipeline:
    """Unified augmentation pipeline for local-region, object-level, and exchange branches.

    Synced transforms (CYWS-style):
        - RandomResizedCrop: same crop region for both views
        - HorizontalFlip / VerticalFlip: synced flips

    Independent transforms:
        - RandomAffine: simulate different viewpoints
        - ColorJitter: simulate different lighting
    """

    def __init__(
        self,
        mode,
        target_size=256,
        *,
        crop_p: float = 0.0,
        crop_independent: bool = False,
        crop_scale: tuple[float, float] = (0.92, 1.0),
        crop_ratio: tuple[float, float] = (0.95, 1.05),
        hflip_p: float = 0.0,
        vflip_p: float = 0.0,
        affine_degrees: float = 30.0,
        affine_translate: tuple[float, float] = (0.1, 0.1),
        affine_scale: tuple[float, float, float, float] = (0.9, 1.1, 0.9, 1.1),
        affine_p: float = 1.0,
        color_jitter: tuple[float, float, float, float] = (0.08, 0.08, 0.08, 0.05),
        color_jitter_p: float = 1.0,
    ):
        super().__init__()
        self.mode = mode
        if isinstance(target_size, (tuple, list)):
            self.target_size = int(target_size[0])
        else:
            self.target_size = int(target_size)

        # Geometric transforms
        # Resize is deferred to dataset return; crop is done here via pure Torch to avoid Kornia 0.6.3 RRC issues
        self.crop_p = float(crop_p)
        self.crop_independent = bool(crop_independent)
        self.crop_scale = (float(crop_scale[0]), float(crop_scale[1]))
        self.crop_ratio = (float(crop_ratio[0]), float(crop_ratio[1]))
        self.hflip_p = float(hflip_p)
        self.vflip_p = float(vflip_p)

        self.aff = K.augmentation.RandomAffine(
            degrees=float(affine_degrees),
            translate=(float(affine_translate[0]), float(affine_translate[1])),
            scale=(
                float(affine_scale[0]),
                float(affine_scale[1]),
                float(affine_scale[2]),
                float(affine_scale[3]),
            ),
            padding_mode="border",
            p=float(affine_p),
            return_transform=True,
            keepdim=True,
        )

        # Color jitter (independent per view; no grayscale/blur)
        b, c, s, h = color_jitter
        self.jit = K.augmentation.ColorJitter(
            float(b), float(c), float(s), float(h), p=float(color_jitter_p), keepdim=True
        )
        # Default: color before geometry; switch to geometry-first when crop/flip is enabled
        self._use_color_after_geometry = bool(
            float(crop_p) > 0.0 or float(hflip_p) > 0.0 or float(vflip_p) > 0.0
        )

    def _sample_crop_params(self, h: int, w: int, device: torch.device):
        """Sample a crop box (top, left, crop_h, crop_w)."""
        area = float(h * w)
        scale = float(torch.empty(1, device=device).uniform_(self.crop_scale[0], self.crop_scale[1]).item())
        ratio = float(torch.empty(1, device=device).uniform_(self.crop_ratio[0], self.crop_ratio[1]).item())

        target_area = area * scale
        crop_w = int(round((target_area * ratio) ** 0.5))
        crop_h = int(round((target_area / ratio) ** 0.5))

        crop_w = max(1, min(crop_w, w))
        crop_h = max(1, min(crop_h, h))

        left = 0 if w == crop_w else int(torch.randint(0, w - crop_w + 1, (1,), device=device).item())
        top = 0 if h == crop_h else int(torch.randint(0, h - crop_h + 1, (1,), device=device).item())
        return top, left, crop_h, crop_w

    def _crop_matrix(self, *, top: int, left: int, crop_h: int, crop_w: int, out_h: int, out_w: int, device):
        """3x3 homogeneous matrix: crop (top,left,crop_h,crop_w) then resize to (out_h,out_w)."""
        sx = float(out_w) / float(crop_w)
        sy = float(out_h) / float(crop_h)
        tx = -float(left) * sx
        ty = -float(top) * sy
        m = torch.tensor([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        return m.unsqueeze(0)

    def _hflip_matrix(self, w: int, device):
        m = torch.tensor([[-1.0, 0.0, float(w - 1)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device)
        return m.unsqueeze(0)

    def _vflip_matrix(self, h: int, device):
        m = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, float(h - 1)], [0.0, 0.0, 1.0]], device=device)
        return m.unsqueeze(0)

    def apply_geometric_sync(self, image1, image2, mask):
        """Apply synced geometric transforms (CYWS-style).

        Workflow: optional RRC → optional flip → independent Affine → CYWS intersection alignment.
        Returns (image1, image2, mask1, mask2, t1, t2).
        """
        # Ensure 4D NCHW
        if len(image1.shape) == 3:
            image1 = image1.unsqueeze(0)
        if len(image2.shape) == 3:
            image2 = image2.unsqueeze(0)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(0)

        if self.mode == "identity":
            identity = torch.eye(3, device=image1.device).unsqueeze(0)
            m = mask.squeeze(0)
            return image1.squeeze(0), image2.squeeze(0), m, m, identity, identity

        # Default: color before geometry (jitter then affine)
        if not self._use_color_after_geometry:
            image1 = self.jit(image1)
            image2 = self.jit(image2)

        original_mask = mask.float()
        original_hw = image1.shape[-2:]

        t1_pre = torch.eye(3, device=image1.device).unsqueeze(0)
        t2_pre = torch.eye(3, device=image2.device).unsqueeze(0)

        # RandomResizedCrop (synced or independent)
        do_crop = torch.rand(1, device=image1.device).item() < self.crop_p
        if do_crop:
            out_h = self.target_size
            out_w = self.target_size
            h, w = int(original_hw[0]), int(original_hw[1])

            if self.crop_independent:
                top1, left1, ch1, cw1 = self._sample_crop_params(h, w, image1.device)
                top2, left2, ch2, cw2 = self._sample_crop_params(h, w, image2.device)
                c1 = self._crop_matrix(top=top1, left=left1, crop_h=ch1, crop_w=cw1, out_h=out_h, out_w=out_w, device=image1.device)
                c2 = self._crop_matrix(top=top2, left=left2, crop_h=ch2, crop_w=cw2, out_h=out_h, out_w=out_w, device=image2.device)
                image1 = K.geometry.transform.warp_affine(image1, c1[:, :2, :], (out_h, out_w), mode="bilinear", padding_mode="border", align_corners=False)
                image2 = K.geometry.transform.warp_affine(image2, c2[:, :2, :], (out_h, out_w), mode="bilinear", padding_mode="border", align_corners=False)
                t1_pre = c1 @ t1_pre
                t2_pre = c2 @ t2_pre
            else:
                top, left, ch, cw = self._sample_crop_params(h, w, image1.device)
                c = self._crop_matrix(top=top, left=left, crop_h=ch, crop_w=cw, out_h=out_h, out_w=out_w, device=image1.device)
                image1 = K.geometry.transform.warp_affine(image1, c[:, :2, :], (out_h, out_w), mode="bilinear", padding_mode="border", align_corners=False)
                image2 = K.geometry.transform.warp_affine(image2, c[:, :2, :], (out_h, out_w), mode="bilinear", padding_mode="border", align_corners=False)
                t1_pre = c @ t1_pre
                t2_pre = c @ t2_pre

        cur_hw = image1.shape[-2:]

        # Synced flips
        if torch.rand(1).item() < self.hflip_p:
            image1 = K.geometry.hflip(image1)
            image2 = K.geometry.hflip(image2)
            t = self._hflip_matrix(int(cur_hw[1]), image1.device)
            t1_pre = t @ t1_pre
            t2_pre = t @ t2_pre

        if torch.rand(1).item() < self.vflip_p:
            image1 = K.geometry.vflip(image1)
            image2 = K.geometry.vflip(image2)
            t = self._vflip_matrix(int(cur_hw[0]), image1.device)
            t1_pre = t @ t1_pre
            t2_pre = t @ t2_pre

        # Independent Affine
        image_shape_as_hw = image1.shape[-2:]
        image1, transform1 = self.aff(image1)
        image2, transform2 = self.aff(image2)

        # CYWS strict alignment: back-project both masks to original coords, intersect, then re-warp
        t1 = transform1 @ t1_pre
        t2 = transform2 @ t2_pre

        mask1_t = apply_transformation_to_mask(original_mask, t1, image_shape_as_hw)
        mask2_t = apply_transformation_to_mask(original_mask, t2, image_shape_as_hw)

        inv1 = torch.inverse(t1)
        inv2 = torch.inverse(t2)
        mask1_inv = apply_transformation_to_mask(mask1_t, inv1, original_hw)
        mask2_inv = apply_transformation_to_mask(mask2_t, inv2, original_hw)

        intersection = (mask1_inv > 0.5).float() * (mask2_inv > 0.5).float()

        mask1_final = apply_transformation_to_mask(intersection, t1, image_shape_as_hw)
        mask2_final = apply_transformation_to_mask(intersection, t2, image_shape_as_hw)

        return (
            image1.squeeze(0),
            image2.squeeze(0),
            mask1_final.squeeze(0),
            mask2_final.squeeze(0),
            t1,
            t2,
        )

    def apply_color_independent(self, image1, image2):
        """Apply independent color jitter to both images."""
        image1 = self.jit(image1)
        image2 = self.jit(image2)

        if len(image1.shape) == 4:
            image1 = image1.squeeze(0)
        if len(image2.shape) == 4:
            image2 = image2.squeeze(0)

        return image1, image2

    def forward_object_exchange(self, base_image, source_image, source_mask):
        """Object Exchange (Copy-Paste): paste source mask region onto base image, then synced geometry."""
        exchanged_image = base_image.clone()

        if len(source_mask.shape) == 2:
            source_mask_3ch = source_mask.unsqueeze(0).expand_as(base_image)
        elif source_mask.shape[0] == 1:
            source_mask_3ch = source_mask.expand_as(base_image)
        else:
            source_mask_3ch = source_mask

        exchanged_image = torch.where(
            source_mask_3ch > 0.5,
            source_image,
            base_image
        )

        exchanged_image, base_image, mask_ex, mask_base, transform1, _ = self.apply_geometric_sync(
            exchanged_image, base_image, source_mask.float()
        )
        if self._use_color_after_geometry:
            exchanged_image, base_image = self.apply_color_independent(exchanged_image, base_image)
        return exchanged_image, base_image, mask_ex, mask_base, transform1

    def forward_object_level(self, image1, image2, mask):
        """Object-Level Change (LaMa / Copy-Paste): synced geometry + independent color."""
        image1, image2, mask1, mask2, transform1, transform2 = self.apply_geometric_sync(image1, image2, mask)
        if self._use_color_after_geometry:
            image1, image2 = self.apply_color_independent(image1, image2)
        return image1, image2, mask1, mask2, transform1

    def forward_local_region(self, image1, image2, mask):
        """Local-Region Change (DTD / Perlin): synced geometry + independent color."""
        image1, image2, mask1, mask2, transform1, _ = self.apply_geometric_sync(image1, image2, mask)
        if self._use_color_after_geometry:
            image1, image2 = self.apply_color_independent(image1, image2)
        return image1, image2, mask1, mask2, transform1

    def forward(self, image1_tensor, image2_tensor, change_mask):
        """Default forward using object-level strategy."""
        return self.forward_object_level(image1_tensor, image2_tensor, change_mask)


def apply_transformation_to_mask(mask_tensor, transformation_matrix, image_shape_as_hw):
    """Apply a transformation matrix to a mask."""
    while len(mask_tensor.shape) < 4:
        mask_tensor = mask_tensor.unsqueeze(0)

    # Kornia RandomAffine returns 3x3 homogeneous; warp_affine expects 2x3
    if transformation_matrix.dim() == 3:
        if transformation_matrix.shape[-2:] == (3, 3):
            transformation_matrix = transformation_matrix[:, :2, :]
    elif transformation_matrix.dim() == 2:
        transformation_matrix = transformation_matrix.unsqueeze(0)
        if transformation_matrix.shape[-2:] == (3, 3):
            transformation_matrix = transformation_matrix[:, :2, :]

    mask_transformed = K.geometry.transform.warp_affine(
        mask_tensor.float(),
        transformation_matrix,
        image_shape_as_hw,
        mode="nearest",
    )
    while len(mask_transformed.shape) > len(mask_tensor.shape):
        mask_transformed = mask_transformed.squeeze(0)
    return mask_transformed
