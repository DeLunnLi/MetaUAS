from copy import deepcopy

import kornia as K
import shapely.affinity
import shapely.geometry
import torch
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon


def make_valid_polygon(shapely_object):
    """Return a valid shapely Polygon; take convex hull of MultiPolygon union, or return empty Polygon."""
    valid_object = make_valid(shapely_object)
    if valid_object.is_empty:
        return valid_object
    if valid_object.geom_type in ["MultiPolygon", "GeometryCollection"]:
        multi_polygon = MultiPolygon()
        for obj in valid_object.geoms:
            if obj.geom_type == "Polygon":
                multi_polygon = multi_polygon.union(obj)
        valid_object = multi_polygon.convex_hull
    if valid_object.geom_type != "Polygon":
        valid_object = Polygon()
    return valid_object


def get_bboxes_as_shapely_objects_from_coco_annotations(annotations):
    """Return bboxes as shapely objects from COCO annotations."""
    shapely_objects = []
    for obj in annotations:
        shapely_objects.append(shapely.geometry.box(*obj["bbox"]))
    return shapely_objects


def get_segmentation_as_shapely_polygons_from_coco_annotations(annotations):
    """Return segmentations as shapely polygons from COCO annotations."""
    shapely_objects = []
    for obj in annotations:
        shapely_object = shapely.geometry.MultiPolygon()
        for obj_segment in obj["segmentation"]:
            try:
                x_coords = obj_segment[0::2]
                y_coords = obj_segment[1::2]
                coordinates = []
                for x, y in zip(x_coords, y_coords):
                    try:
                        coordinates.append([float(x), float(y)])
                    except (ValueError, TypeError):
                        continue
                if coordinates:
                    shapely_object = shapely_object.union(shapely.geometry.Polygon(coordinates))
            except Exception:
                continue

        # Convert to valid polygon
        shapely_polygon = make_valid_polygon(shapely_object)
        shapely_objects.append(shapely_polygon)

    return shapely_objects


def convert_shapely_polygon_into_coco_segmentation(shapely_object):
    """Return shapely polygon in COCO segmentation format."""
    segmentation = []
    for point in shapely_object.exterior.coords[:-1]:
        segmentation.append(point[0])
        segmentation.append(point[1])
    return [segmentation]


def get_axis_aligned_bbox_for_segmentation(shapely_object):
    """Compute axis-aligned bounding box for a segmentation polygon."""
    x_coords = [x for x, y in shapely_object.exterior.coords]
    y_coords = [y for x, y in shapely_object.exterior.coords]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


def apply_kornia_transformation_to_shapely_objects(
    shapely_objects, transformation, image_shape_as_hw, keep_empty=False
):
    """Apply a 2D affine transformation to shapely polygon objects."""

    transformation = transformation.squeeze().reshape(-1)
    transformation_parameters = torch.index_select(
        transformation, 0, torch.tensor([0, 1, 3, 4, 2, 5])
    )
    image_bounding_box = shapely.geometry.box(0, 0, *image_shape_as_hw[::-1])
    is_a_single_object = isinstance(shapely_objects, shapely.geometry.Polygon)
    if is_a_single_object:
        shapely_objects = [shapely_objects]
    transformed_objects = []
    for obj in shapely_objects:
        transformed_obj = make_valid_polygon(
            shapely.affinity.affine_transform(obj, transformation_parameters)
        )
        visible_obj = make_valid_polygon(transformed_obj.intersection(image_bounding_box))
        if not visible_obj.is_empty or keep_empty:
            transformed_objects.append(visible_obj)
    if is_a_single_object and len(transformed_objects) == 0:
        transformed_objects.append(shapely.geometry.Polygon())
    return transformed_objects if not is_a_single_object else transformed_objects[0]


def merge_shapely_polygons_into_annotations(transformed_segmentations, annotations):
    """Update COCO annotations with geometrically transformed segmentations."""
    transformed_annotations = []
    for shapely_object, annotation in zip(transformed_segmentations, annotations):
        annotation_copy = deepcopy(annotation)
        if shapely_object.is_empty or shapely_object.geom_type != "Polygon":
            continue
        annotation_copy["segmentation"] = convert_shapely_polygon_into_coco_segmentation(
            shapely_object
        )
        annotation_copy["bbox"] = get_axis_aligned_bbox_for_segmentation(shapely_object)
        transformed_annotations.append(annotation_copy)
    return transformed_annotations


def resize_image_and_annotations(input_image_as_tensor, output_shape_as_hw, annotations=None):
    """Resize image and optionally its annotations to the specified output dimensions."""
    resize_augmentation = K.augmentation.Resize(output_shape_as_hw, return_transform=True)
    resized_input_tensor, transformation = resize_augmentation(input_image_as_tensor)
    resized_input_tensor = resized_input_tensor.squeeze()
    if annotations is None:
        return resized_input_tensor, None
    segmentations_as_shapely_objects = get_segmentation_as_shapely_polygons_from_coco_annotations(
        annotations
    )
    transformed_segmentations = apply_kornia_transformation_to_shapely_objects(
        segmentations_as_shapely_objects,
        transformation,
        resized_input_tensor.shape[-2:],
    )
    transformed_annotations = merge_shapely_polygons_into_annotations(
        transformed_segmentations, annotations
    )
    return resized_input_tensor, transformed_annotations
