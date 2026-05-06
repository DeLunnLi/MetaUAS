from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class MvtecDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        CLASS_NAME,
        resize_shape=(256, 256),
        normal_image_override: Optional[str] = None,
    ):
        self.dataset_path = dataset_path
        self.CLASS_NAME = CLASS_NAME
        self.normal_image_override = normal_image_override
        self.test_img_paths, self.test_label_paths, self.test_mask_paths = self.load_test_dataset()

        self.transform_image = T.Compose([
            T.Resize((256,256), Image.LANCZOS),
            T.CenterCrop(256),
            T.ToTensor()])

    def __len__(self):
        return len(self.test_img_paths)

    def get_len(self):
        return len(self.test_img_paths)

    def __getitem__(self,idx):
        test_img_path,test_label = self.test_img_paths[idx], self.test_label_paths[idx]
        class_name = self.CLASS_NAME
        image = Image.open(test_img_path)
        if class_name in ['zipper', 'screw', 'grid']:
            image = np.expand_dims(np.array(image), axis=2)
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        image = self.transform_image(image)

        return image, test_label

    def get_random_normal_image(self):
        if self.normal_image_override:
            random_image_path = self.normal_image_override
            if not os.path.isfile(random_image_path):
                raise FileNotFoundError(f"指定的正常样本不存在: {random_image_path}")
        else:
            normal_image_path = os.path.join(self.dataset_path, self.CLASS_NAME, "train", "good")
            train_img_list = sorted(
                [
                    os.path.join(normal_image_path, f)
                    for f in os.listdir(normal_image_path)
                    if f.endswith(".png")
                ]
            )
            random_image_path = random.choice(train_img_list)
        class_name = self.CLASS_NAME
        image = Image.open(random_image_path)
        if class_name in ['zipper', 'screw', 'grid']:
            image = np.expand_dims(np.array(image), axis=2)
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        image = self.transform_image(image)
        return image

    def load_test_dataset(self):
        
        test_img_paths, test_labels, test_mask_paths = [], [], []
        test_img_folder = os.path.join(self.dataset_path, self.CLASS_NAME, 'test')
        test_mask_folder = os.path.join(self.dataset_path, self.CLASS_NAME, 'ground_truth')

        test_img_types = sorted(os.listdir(test_img_folder))

        for test_img_type in test_img_types:
            test_img_dir = os.path.join(test_img_folder, test_img_type)
            if not os.path.isdir(test_img_dir):
                continue
            test_img_list = sorted([os.path.join(test_img_dir, f)
                                     for f in os.listdir(test_img_dir)
                                     if f.endswith('.png')])
            test_img_paths.extend(test_img_list)

            if test_img_type == 'good':
                test_labels.extend([0] * len(test_img_list))
                test_mask_paths.extend([None] * len(test_img_list))
            else:
                test_labels.extend([1] * len(test_img_list))
                test_mask_dir = os.path.join(test_mask_folder, test_img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in test_img_list]
                test_mask_list_one_type = [os.path.join(test_mask_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                test_mask_paths.extend(test_mask_list_one_type)

        return test_img_paths, test_labels, test_mask_paths