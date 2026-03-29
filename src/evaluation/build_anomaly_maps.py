import os
import shutil
import glob
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter

from src.utils.inference import *
from src.utils.anomaly_maps import *

subpath_list = os.getcwd().split(os.sep)
root_directory_path = os.path.join(*subpath_list[:-1])
current_dir = subpath_list[-1]

###  To modify these configuration values
DATASET_NAME = "BraTS2021_slice"
SPLIT = "valid"
# How many random target boxes to sample per image
NUM_TARGET_BOXES_PER_IMAGE = 30
AGGREGATE_PATCHES_STRATEGY = "max"
GAUSSIAN_FILTER_BLUR_SIGMA = 4
DISTANCE = "l2"
EXPERIMENT_FOLDER_NAME = "test_experiment"
MODEL_INPUT_CROP_SIZE = 224
PATCH_SIZE = 16
###

EVALUATION_DATASET_PATH = f"/media/vhondru/hdd/istoian/BMAD/dataset/{DATASET_NAME}/{SPLIT}"

EXPERIMENT_FOLDER_PATH = (
    f"/{root_directory_path}/{current_dir}/"
    f"experiments_{DATASET_NAME}/{EXPERIMENT_FOLDER_NAME}/results_{SPLIT}"
)

ANOMALY_MAPS_ARRAYS_PATH = (
    f"{EXPERIMENT_FOLDER_PATH}/"
    f"anomaly_maps_arrays_{DISTANCE}_distance_{AGGREGATE_PATCHES_STRATEGY}_agg_strategy_"
    f"{NUM_TARGET_BOXES_PER_IMAGE}_boxes_sigma_{GAUSSIAN_FILTER_BLUR_SIGMA}"
)

ANOMALY_MAPS_IMAGES_PATH = (
    f"{EXPERIMENT_FOLDER_PATH}/"
    f"anomaly_maps_images_{DISTANCE}_distance_{AGGREGATE_PATCHES_STRATEGY}_agg_strategy_"
    f"{NUM_TARGET_BOXES_PER_IMAGE}_boxes_sigma_{GAUSSIAN_FILTER_BLUR_SIGMA}"
)

SAVED_DICTS_FOLDER = (
    f"{EXPERIMENT_FOLDER_PATH}/"
    f"image_per_patch_{DISTANCE}_distances_{NUM_TARGET_BOXES_PER_IMAGE}_seeds"
)

SAVE_CROPPED_BORDERS_DICT_FOLDER = (
    f"{EXPERIMENT_FOLDER_PATH}/"
    "cropped_borders_dicts/"
)

if not os.path.exists(ANOMALY_MAPS_IMAGES_PATH):
    os.makedirs(ANOMALY_MAPS_IMAGES_PATH, exist_ok=True)

if not os.path.exists(ANOMALY_MAPS_ARRAYS_PATH):
    os.makedirs(ANOMALY_MAPS_ARRAYS_PATH, exist_ok=True)

def get_dataset_anomaly_map_min_max(results_dicts_file_paths):
    # Compute the min, max statistics at the dataset level
    dataset_min = np.inf
    dataset_max = -np.inf

    print("Computing dataset min/max values of the generated anomaly maps ...")
    for results_dicts_file_path in tqdm(results_dicts_file_paths):
        with open(results_dicts_file_path, "r") as f:
            distance_dict = json.load(f)

        saved_borders_dict_path = results_dicts_file_path \
                .replace("_distances_label_", "_cropped_borders_dict_label_") \
                .replace(SAVED_DICTS_FOLDER, SAVE_CROPPED_BORDERS_DICT_FOLDER)
        with open(saved_borders_dict_path, "r") as f:
            borders_dict = json.load(f)

        crop_height = borders_dict["bottom"] - borders_dict["top"] + 1
        crop_width = borders_dict["right"] - borders_dict["left"] + 1
        
        anomaly_map = build_anomaly_map_interpolation(
            per_patch_distances_dict=distance_dict,
            model_input_height=MODEL_INPUT_CROP_SIZE,
            model_input_width=MODEL_INPUT_CROP_SIZE,
            crop_height=crop_height,
            crop_width=crop_width,
            patch_size=PATCH_SIZE,
            agg_strategy=AGGREGATE_PATCHES_STRATEGY)
        
        if GAUSSIAN_FILTER_BLUR_SIGMA != -1:
            anomaly_map = gaussian_filter(anomaly_map, GAUSSIAN_FILTER_BLUR_SIGMA)

        dataset_min = min(np.min(anomaly_map), dataset_min)
        dataset_max = max(np.max(anomaly_map), dataset_max)
    
    return dataset_min, dataset_max

def main():
    results_dicts_file_paths = glob.glob(os.path.join(SAVED_DICTS_FOLDER, "*.json"))
    print("Number of samples to evaluate on: ", len(results_dicts_file_paths))
    dataset_min, dataset_max = get_dataset_anomaly_map_min_max(results_dicts_file_paths=results_dicts_file_paths)
        
    print("Building the anomaly maps...")
    for results_dicts_file_path in tqdm(results_dicts_file_paths):
        image_name = Path(results_dicts_file_path).stem.split("_distances_label")[0]
        label = int(results_dicts_file_path[-6])

        saved_borders_dict_path = results_dicts_file_path \
            .replace("_distances_label_", "_cropped_borders_dict_label_") \
            .replace(SAVED_DICTS_FOLDER, SAVE_CROPPED_BORDERS_DICT_FOLDER)

        with open(results_dicts_file_path, "r") as f:
            results_distances_dict = json.load(f)

        with open(saved_borders_dict_path, "r") as f:
            borders_dict = json.load(f)

        # TODO - why scaling the borders using the min, max of the anomaly map without borders
        anomaly_map = build_anomaly_map_with_borders(
            per_patch_distances_dict=results_distances_dict,
            patch_size=PATCH_SIZE,
            borders_dict=borders_dict,
            model_input_height=MODEL_INPUT_CROP_SIZE,
            model_input_width=MODEL_INPUT_CROP_SIZE,
            original_image_height=borders_dict["original_image_height"],
            original_image_width=borders_dict["original_image_width"],
            agg_strategy=AGGREGATE_PATCHES_STRATEGY,
            interpolation_method=cv2.INTER_LINEAR
        )

        if GAUSSIAN_FILTER_BLUR_SIGMA != -1:
            anomaly_map = gaussian_filter(anomaly_map, GAUSSIAN_FILTER_BLUR_SIGMA)

        # MIN-MAX scaling to convert the pixels in the [0, 1] range so that we can display the anomaly maps
        # The min and the max values are computed at dataset level
        scaled_anomaly_map =  (anomaly_map - dataset_min) / (dataset_max - dataset_min + 1e-8)

        save_results(image_name, label, scaled_anomaly_map)

def save_results(image_name, label, scaled_anomaly_map):
    # Save the scaled anomaly map
    scaled_anomaly_map_path = f"{ANOMALY_MAPS_ARRAYS_PATH}/{image_name}_anomaly_map_label_{label}.npy"
    np.save(file=scaled_anomaly_map_path, arr=scaled_anomaly_map)   
    Image.fromarray((255 * scaled_anomaly_map) \
        .astype("uint8")) \
        .convert("L") \
        .save(f"{ANOMALY_MAPS_IMAGES_PATH}/{image_name}_anomaly_map_label_{label}.png")

    # Convert the anomaly map into a heatmap and superimpose it on the initial image
    scaled_anomaly_map = (scaled_anomaly_map * 255).astype(np.uint8)
    scaled_anomaly_map = cvt2heatmap(scaled_anomaly_map)

    label_subpath = "good" if label == 0 else "ungood"
    image_path = os.path.join(EVALUATION_DATASET_PATH, label_subpath, "img", image_name + ".png")
    initial_image = cv2.imread(image_path)
    cv2.imwrite(f"{ANOMALY_MAPS_IMAGES_PATH}/{image_name}_original.png", initial_image)
    initial_image = (initial_image * 255).astype(np.uint8)

    heatmap = cv2.addWeighted(scaled_anomaly_map, 0.4, initial_image, (1 - 0.4), 0)
    cv2.imwrite(f"{ANOMALY_MAPS_IMAGES_PATH}/{image_name}_heatmap.png", heatmap)

    # We are saving in the same folder the ground-truth segmentation mask for comparing it with the generated anomaly map
    
    source_segmentation_mask_path = os.path.join(EVALUATION_DATASET_PATH, label_subpath, "label", image_name + ".png")
    segmentation_image = Image.open(source_segmentation_mask_path)
    segmentation_arr = np.array(segmentation_image)

    # Preprocessing the segmentation masks: from BMAD codebase: https://github.com/DorisBao/BMAD/blob/main/UTRAD/data.py#L130 !!!
    segmentation_arr[segmentation_arr >= 0.1] = 1
    segmentation_arr[segmentation_arr < 0.1] = 0
 
    # Saving the image for visualization
    dest_segmentation_mask_path = f"{ANOMALY_MAPS_IMAGES_PATH}/{image_name}_segmentation_label_{label}.png"
    Image.fromarray((255 * segmentation_arr) \
        .astype("uint8")) \
        .convert("L") \
        .save(dest_segmentation_mask_path)

    # Saving the segmentation map as a numpy array for computing the pixel-level metrics
    dest_segmentation_arr_path = f"{ANOMALY_MAPS_ARRAYS_PATH}/{image_name}_segmentation_label_{label}.npy"
    np.save(file=dest_segmentation_arr_path, arr=segmentation_arr)

if __name__ == "__main__":
    main()
