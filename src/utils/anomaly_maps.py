import numpy as np
import cv2
from typing import Dict, List


def build_patches_anomaly_map(
        per_patch_distances_dict: Dict[int, List[float]],
        model_input_height: int,
        model_input_width: int,
        patch_size: int,
        agg_strategy: str= "mean"
    ):

    assert model_input_height % patch_size == 0
    assert model_input_width % patch_size == 0
    
    num_patches_height = model_input_height // patch_size
    num_patches_width = model_input_width // patch_size

    patches_anomaly_map = np.zeros((num_patches_height, num_patches_width))
    for patch_idx, distances in per_patch_distances_dict.items():
        if len(distances) > 0:
            if agg_strategy == "mean":
                aggregated_distance_per_patch = np.mean(distances)
            elif agg_strategy == "max":
                aggregated_distance_per_patch = np.max(distances)
            else:
                raise ValueError(f"Unsupported aggregation strategy: {agg_strategy}")

            row = int(patch_idx) // num_patches_width
            col = int(patch_idx) % num_patches_width

            patches_anomaly_map[row, col] = aggregated_distance_per_patch

    return patches_anomaly_map


def build_anomaly_map_with_borders(
        per_patch_distances_dict: Dict[int, List[float]],
        patch_size: int,
        borders_dict: Dict[str, int],
        model_input_height: int,
        model_input_width: int,
        original_image_height: int,
        original_image_width: int,
        agg_strategy: str= "mean",
        interpolation_method=cv2.INTER_LINEAR
    ):

    crop_height = borders_dict["bottom"] - borders_dict["top"] + 1
    crop_width = borders_dict["right"] - borders_dict["left"] + 1

    resized_crop_anomaly_map = build_anomaly_map_interpolation(
        per_patch_distances_dict=per_patch_distances_dict,
        model_input_height=model_input_height,
        model_input_width=model_input_width,
        crop_height=crop_height,
        crop_width=crop_width,
        patch_size=patch_size,
        agg_strategy=agg_strategy,
        interpolation_method=interpolation_method
    )

    assert resized_crop_anomaly_map.shape == (
        borders_dict["bottom"] - borders_dict["top"] + 1, 
        borders_dict["right"] - borders_dict["left"] + 1
    )

    anomaly_map_with_borders = np.zeros(
        shape=(original_image_height, original_image_width),
        dtype=resized_crop_anomaly_map.dtype,
    )

    anomaly_map_with_borders[
        borders_dict["top"]: borders_dict["bottom"] + 1,
        borders_dict["left"]: borders_dict["right"] + 1
    ] = resized_crop_anomaly_map

    return anomaly_map_with_borders


def build_anomaly_map_repeat(
        per_patch_distances_dict: Dict[int, List[float]],
        model_input_height: int,
        model_input_width: int,
        patch_size: int,
        agg_strategy: str= "mean"
    ):
    num_patches_width = model_input_width // patch_size

    anomaly_map = np.zeros((model_input_height, model_input_width))
    for patch_idx, distances in per_patch_distances_dict.items():
        if len(distances) > 0:
            if agg_strategy == "mean":
                aggregated_distance_per_patch = np.mean(distances)
            else:
                aggregated_distance_per_patch = np.max(distances)

            row = int(patch_idx) // num_patches_width
            col = int(patch_idx) % num_patches_width
            
            anomaly_map[
                row * patch_size: (row + 1) * patch_size,
                col * patch_size: (col + 1) * patch_size] = aggregated_distance_per_patch
    
    return anomaly_map

def build_anomaly_map_interpolation(
        per_patch_distances_dict: Dict[int, List[float]],
        model_input_height: int,
        model_input_width: int,
        crop_height: int,
        crop_width: int,
        patch_size: int,
        agg_strategy: str= "mean",
        interpolation_method=cv2.INTER_LINEAR
    ):
    # 1. Generate the low-resolution (at patch level) anomaly map grid (14x14)
    patches_anomaly_map = build_patches_anomaly_map(
        per_patch_distances_dict=per_patch_distances_dict,
        model_input_height=model_input_height,
        model_input_width=model_input_width,
        patch_size=patch_size,
        agg_strategy=agg_strategy
    )

    # 2. Resize to the model's input resolution first (e.g 224x224 for BraTS_slice)
    # This ensures the patch values fill their respective tiles before scaling to the final crop.
    model_res_map = cv2.resize(
        patches_anomaly_map, 
        dsize=(model_input_width, model_input_height), 
        interpolation=cv2.INTER_NEAREST
    )

    # 3. Smooth and Resize to the actual crop dimensions
    # cv2.resize expects (width, height)
    anomaly_map = cv2.resize(
        model_res_map, 
        dsize=(crop_width, crop_height), 
        interpolation=interpolation_method
    )

    return anomaly_map

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return heatmap
