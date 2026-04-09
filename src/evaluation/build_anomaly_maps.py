import os
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter

from src.utils.inference import *
from src.utils.configs import *
from src.utils.anomaly_maps import *

STORAGE_PARENT_FOLDER_PATH = "/media/vhondru/hdd/istoian/BMAD/dataset/"


class AnomalyMapsBuilder:
    def __init__(self, inference_config_path: str, use_distillation_model: bool=False):
        self.inference_config = read_inference_anomaly_detection_config(config_path=inference_config_path)

        subpath_list = os.getcwd().split(os.sep)
        root_directory_path = os.path.join(*subpath_list[:-1])
        current_dir = subpath_list[-1]

        self.evaluation_dataset_path = (f"{STORAGE_PARENT_FOLDER_PATH}"
                                        f"{self.inference_config.dataset_name}/{self.inference_config.split}")
        
        self.experiment_folder_path = (
            f"/{root_directory_path}/{current_dir}/"
            f"experiments_{self.inference_config.dataset_name}/"
            f"{self.inference_config.experiment_name}/results_{self.inference_config.split}"
        )

        self.train_config_path = (
            f"/{root_directory_path}/{current_dir}/"
            f"experiments_{self.inference_config.dataset_name}/"
            f"{self.inference_config.experiment_name}/logs/config_run.yaml"
        )

        # Read the ijepa training configuration file (which was used for training the checkpoint)
        self.train_config = read_train_ijepa_config(config_path=self.train_config_path)

        if not use_distillation_model:
            # computing the anomaly map using the random boxes sampling
            self.anomaly_maps_arrays_path = (
                f"{self.experiment_folder_path}/"
                f"anomaly_maps_arrays_{self.inference_config.distance}_distance_"
                f"{self.inference_config.aggregate_patches_strategy}_agg_strategy_"
                f"{self.inference_config.num_sampled_random_boxes_per_image}_boxes_sigma_"
                f"{self.inference_config.gaussian_filter_blur_sigma}"
            )

            self.anomaly_maps_images_path = (
                f"{self.experiment_folder_path}/"
                f"anomaly_maps_images_{self.inference_config.distance}_distance_"
                f"{self.inference_config.aggregate_patches_strategy}_agg_strategy_"
                f"{self.inference_config.num_sampled_random_boxes_per_image}_boxes_"
                f"sigma_{self.inference_config.gaussian_filter_blur_sigma}"
            )

            self.saved_dicts_folder = (
                f"{self.experiment_folder_path}/"
                f"image_per_patch_{self.inference_config.distance}_distances_"
                f"{self.inference_config.num_sampled_random_boxes_per_image}_seeds"
            )
        else:
            # computing the anomaly boxes using the distillation model output
            self.anomaly_maps_arrays_path = (
                f"{self.experiment_folder_path}/"
                f"anomaly_maps_arrays_distillation_{self.inference_config.distance}_distance_"
                f"sigma_{self.inference_config.gaussian_filter_blur_sigma}"
            )

            self.anomaly_maps_images_path = (
                f"{self.experiment_folder_path}/"
                f"anomaly_maps_images_distillation_{self.inference_config.distance}_distance_"
                f"sigma_{self.inference_config.gaussian_filter_blur_sigma}"
            )
            self.saved_dicts_folder = (
                f"{self.experiment_folder_path}/"
                f"image_per_patch_{self.inference_config.distance}_distances_"
                f"distilled_model/"
            )

        self.save_cropped_borders_dict_folder = (
            f"{self.experiment_folder_path}/"
            "cropped_borders_dicts/"
        )

        if not os.path.exists(self.anomaly_maps_images_path):
            os.makedirs(self.anomaly_maps_images_path, exist_ok=True)

        if not os.path.exists(self.anomaly_maps_arrays_path):
            os.makedirs(self.anomaly_maps_arrays_path, exist_ok=True)

    def get_dataset_anomaly_map_min_max(self, results_dicts_file_paths):
        # Compute the min, max statistics at the dataset level
        dataset_min = np.inf
        dataset_max = -np.inf

        print("Computing dataset min/max values of the generated anomaly maps ...")
        for results_dicts_file_path in tqdm(results_dicts_file_paths):
            with open(results_dicts_file_path, "r") as f:
                distance_dict = json.load(f)

            saved_borders_dict_path = results_dicts_file_path \
                    .replace("_distances_label_", "_cropped_borders_dict_label_") \
                    .replace(self.saved_dicts_folder, self.save_cropped_borders_dict_folder)
            with open(saved_borders_dict_path, "r") as f:
                borders_dict = json.load(f)

            crop_height = borders_dict["bottom"] - borders_dict["top"] + 1
            crop_width = borders_dict["right"] - borders_dict["left"] + 1
            
            anomaly_map = build_anomaly_map_interpolation(
                per_patch_distances_dict=distance_dict,
                model_input_height=self.train_config.crop_size,
                model_input_width=self.train_config.crop_size,
                crop_height=crop_height,
                crop_width=crop_width,
                patch_size=self.train_config.patch_size,
                agg_strategy=self.inference_config.aggregate_patches_strategy)
            
            if self.inference_config.gaussian_filter_blur_sigma != -1:
                anomaly_map = gaussian_filter(
                    anomaly_map,
                    self.inference_config.gaussian_filter_blur_sigma
                )

            dataset_min = min(np.min(anomaly_map), dataset_min)
            dataset_max = max(np.max(anomaly_map), dataset_max)
        
        return dataset_min, dataset_max
    
    def save_results(self, image_name, label, scaled_anomaly_map):
        # Save the scaled anomaly map
        scaled_anomaly_map_path = (
            f"{self.anomaly_maps_arrays_path}/"
            f"{image_name}_anomaly_map_label_{label}.npy"
        )
        np.save(file=scaled_anomaly_map_path, arr=scaled_anomaly_map)   
        Image.fromarray((255 * scaled_anomaly_map) \
            .astype("uint8")) \
            .convert("L") \
            .save(f"{self.anomaly_maps_images_path}/{image_name}_anomaly_map_label_{label}.png")

        # Convert the anomaly map into a heatmap and superimpose it on the initial image
        scaled_anomaly_map = (scaled_anomaly_map * 255).astype(np.uint8)
        scaled_anomaly_map = cvt2heatmap(scaled_anomaly_map)

        label_subpath = "good" if label == 0 else "ungood"
        image_path = os.path.join(
            self.evaluation_dataset_path,
            label_subpath,
            "img", image_name + ".png"
        )
        initial_image = cv2.imread(image_path)
        cv2.imwrite(f"{self.anomaly_maps_images_path}/{image_name}_original.png", initial_image)
        initial_image = (initial_image * 255).astype(np.uint8)

        heatmap = cv2.addWeighted(scaled_anomaly_map, 0.4, initial_image, (1 - 0.4), 0)
        cv2.imwrite(f"{self.anomaly_maps_images_path}/{image_name}_heatmap.png", heatmap)

        # We are saving in the same folder the ground-truth segmentation mask for comparing it with the generated anomaly map
        source_segmentation_mask_path = os.path.join(
            self.evaluation_dataset_path,
            label_subpath,
            "label",
            image_name + ".png"
        )
        segmentation_image = Image.open(source_segmentation_mask_path).convert("L")
        segmentation_arr = np.array(segmentation_image)

        # Preprocessing the segmentation masks: from BMAD codebase: https://github.com/DorisBao/BMAD/blob/main/UTRAD/data.py#L130 !!!
        segmentation_arr[segmentation_arr >= 0.1] = 1
        segmentation_arr[segmentation_arr < 0.1] = 0
    
        # Saving the image for visualization
        dest_segmentation_mask_path = (
            f"{self.anomaly_maps_images_path}/"
            f"{image_name}_segmentation_mask_label_{label}.png"
        )
        Image.fromarray((255 * segmentation_arr) \
            .astype("uint8")) \
            .convert("L") \
            .save(dest_segmentation_mask_path)

        # Saving the segmentation map as a numpy array for computing the pixel-level metrics
        dest_segmentation_arr_path = (
            f"{self.anomaly_maps_arrays_path}/"
            f"{image_name}_segmentation_mask_label_{label}.npy"
        )
        np.save(file=dest_segmentation_arr_path, arr=segmentation_arr)
   
    def run(self):
        results_dicts_file_paths = glob.glob(os.path.join(self.saved_dicts_folder, "*.json"))
        print("Number of samples to evaluate on: ", len(results_dicts_file_paths))
        dataset_min, dataset_max = self.get_dataset_anomaly_map_min_max(results_dicts_file_paths=results_dicts_file_paths)
            
        print("Building the anomaly maps...")
        for results_dicts_file_path in tqdm(results_dicts_file_paths):
            image_name = Path(results_dicts_file_path).stem.split("_distances_label")[0]
            label = int(results_dicts_file_path[-6])

            saved_borders_dict_path = results_dicts_file_path \
                .replace("_distances_label_", "_cropped_borders_dict_label_") \
                .replace(self.saved_dicts_folder, self.save_cropped_borders_dict_folder)

            with open(results_dicts_file_path, "r") as f:
                results_distances_dict = json.load(f)

            with open(saved_borders_dict_path, "r") as f:
                borders_dict = json.load(f)

            # TODO - why scaling the borders using the min, max of the anomaly map without borders
            anomaly_map = build_anomaly_map_with_borders(
                per_patch_distances_dict=results_distances_dict,
                patch_size=self.train_config.patch_size,
                borders_dict=borders_dict,
                model_input_height=self.train_config.crop_size,
                model_input_width=self.train_config.crop_size,
                original_image_height=borders_dict["original_image_height"],
                original_image_width=borders_dict["original_image_width"],
                agg_strategy=self.inference_config.aggregate_patches_strategy,
                interpolation_method=cv2.INTER_LINEAR
            )

            if self.inference_config.gaussian_filter_blur_sigma != -1:
                anomaly_map = gaussian_filter(anomaly_map, self.inference_config.gaussian_filter_blur_sigma)

            # MIN-MAX scaling to convert the pixels in the [0, 1] range so that we can display the anomaly maps
            # The min and the max values are computed at dataset level
            scaled_anomaly_map =  (anomaly_map - dataset_min) / (dataset_max - dataset_min + 1e-8)

            self.save_results(
                image_name=image_name, 
                label=label,
                scaled_anomaly_map=scaled_anomaly_map
            )

def main(*args, **kwargs):
    parser = argparse.ArgumentParser(
        description=(
            f"Build the anomaly maps using the predictions with on the randomly"
            f" sampled boxes generated with generate_predictions_random_boxes.py script")
    )
    parser.add_argument(
        "--inference_config_path", 
        type=str,
        help="Path to the inference configuration that was used for generating the results."
    )
    parser.add_argument(
        "--use_distillation_model",
        action="store_true",
        help="Enable anomaly map computation using the distillation model"
    )
    args = parser.parse_args()

    anomaly_maps_builder = AnomalyMapsBuilder(
        inference_config_path=args.inference_config_path,
        use_distillation_model=args.use_distillation_model
    )
    anomaly_maps_builder.run()

if __name__ == "__main__":
    main()
