from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


VISA_CLASS_NAMES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]


class VisaDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        class_name: str,
        resize_shape: tuple[int, int] = (256, 256),
        normal_image_override: Optional[str] = None,
    ):
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.normal_image_override = normal_image_override
        self.test_img_paths, self.test_label_paths, self.test_mask_paths = self.load_test_dataset()

        self.transform_image = T.Compose([
            T.Resize(resize_shape, Image.LANCZOS),
            T.CenterCrop(resize_shape[0]),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.test_img_paths)

    def __getitem__(self, idx):
        test_img_path = self.test_img_paths[idx]
        test_label = self.test_label_paths[idx]
        image = Image.open(test_img_path).convert("RGB")
        image = self.transform_image(image)
        mask_path = self.test_mask_paths[idx]
        return image, test_label, mask_path

    def get_random_normal_image(self):
        if self.normal_image_override:
            image = Image.open(self.normal_image_override).convert("RGB")
        else:
            normal_dir = os.path.join(self.dataset_path, self.class_name, "Data", "Images", "Normal")
            normal_list = sorted(
                [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            )
            normal_path = random.choice(normal_list)
            image = Image.open(normal_path).convert("RGB")
        image = self.transform_image(image)
        return image

    def load_test_dataset(self):
        test_img_paths, test_labels, test_mask_paths = [], [], []
        data_root = os.path.join(self.dataset_path, self.class_name, "Data", "Images")

        # Normal (good) images
        normal_dir = os.path.join(data_root, "Normal")
        if os.path.isdir(normal_dir):
            normal_list = sorted([
                os.path.join(normal_dir, f) for f in os.listdir(normal_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            test_img_paths.extend(normal_list)
            test_labels.extend([0] * len(normal_list))
            test_mask_paths.extend([None] * len(normal_list))

        # Anomaly images
        anomaly_dir = os.path.join(data_root, "Anomaly")
        if os.path.isdir(anomaly_dir):
            anomaly_list = sorted([
                os.path.join(anomaly_dir, f) for f in os.listdir(anomaly_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            mask_dir = os.path.join(self.dataset_path, self.class_name, "Data", "Masks", "Anomaly")
            for img_path in anomaly_list:
                test_img_paths.append(img_path)
                test_labels.append(1)
                # Mask has same stem but .png extension
                stem = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = os.path.join(mask_dir, f"{stem}.png")
                if os.path.isfile(mask_path):
                    test_mask_paths.append(mask_path)
                else:
                    # Some VisA masks use _mask suffix
                    alt_mask = os.path.join(mask_dir, f"{stem}_mask.png")
                    test_mask_paths.append(alt_mask if os.path.isfile(alt_mask) else None)

        return test_img_paths, test_labels, test_mask_paths
