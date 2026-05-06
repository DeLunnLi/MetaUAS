from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import yaml
from easydict import EasyDict
from pycocotools import mask as coco_mask_utils

def get_easy_dict_from_yaml_file(path_to_yaml_file):
    """Read a YAML file and return it as an EasyDict."""
    with open(path_to_yaml_file, "r") as stream:
        yaml_file = yaml.safe_load(stream)
    return EasyDict(yaml_file)


def single_coco_annotation_to_mask_image(annotation, image_shape_as_hw):
    """Convert a single COCO annotation (polygon/RLE) to binary mask."""
    h, w = image_shape_as_hw
    segm = annotation["segmentation"]
    if type(segm) == list:
        rles = coco_mask_utils.frPyObjects(segm, h, w)
        rle = coco_mask_utils.merge(rles)
    elif type(segm["counts"]) == list:
        rle = coco_mask_utils.frPyObjects(segm, h, w)
    else:
        rle = annotation["segmentation"]
    m = coco_mask_utils.decode(rle)
    return m


def coco_annotations_to_mask_np_array(list_of_annotations, image_shape_as_hw):
    """Merge a list of COCO annotations into a single binary mask."""
    mask = np.zeros(image_shape_as_hw, dtype=bool)
    for annotation in list_of_annotations:
        object_mask = single_coco_annotation_to_mask_image(annotation, image_shape_as_hw)
        mask = np.maximum(object_mask, mask)
    return mask


def cache_data_triton(path_to_dataset, path_to_file, machine):
    """Cache data to /tmp when running on triton/slurm."""
    if machine not in ["triton", "slurm"]:
        return os.path.join(path_to_dataset, path_to_file)
    caching_location = os.path.join("/tmp/", path_to_file)
    if not os.path.exists(caching_location):
        Path(caching_location).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(path_to_dataset, path_to_file), caching_location)
        os.sync()
    return caching_location
