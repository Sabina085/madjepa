import os
import argparse
import glob
import pandas as pd
from src.utils.configs import *
from src.utils.compute_metrics import compute_roc_auc_metrics

class MetricsComputer:
    def __init__(self, inference_config_path: str):
        self.inference_config = read_inference_anomaly_detection_config(
            config_path=inference_config_path)

        subpath_list = os.getcwd().split(os.sep)
        root_directory_path = os.path.join(*subpath_list[:-1])
        current_dir = subpath_list[-1]

        self.experiment_folder_path = (
            f"/{root_directory_path}/{current_dir}/"
            f"experiments_{self.inference_config.dataset_name}/"
            f"{self.inference_config.experiment_name}/results_{self.inference_config.split}"
        )

        self.anomaly_maps_arrays_path = (
            f"{self.experiment_folder_path}/"
            f"anomaly_maps_arrays_{self.inference_config.distance}_distance_"
            f"{self.inference_config.aggregate_patches_strategy}_agg_strategy_"
            f"{self.inference_config.num_sampled_random_boxes_per_image}_boxes_"
            f"sigma_{self.inference_config.gaussian_filter_blur_sigma}"
        )

        self.metrics_folder = f"{self.experiment_folder_path}/Metrics"
        os.makedirs(self.metrics_folder, exist_ok=True)

        self.metrics_path = (
            f"{self.metrics_folder}/metrics_{self.inference_config.distance}_distance_"
            f"{self.inference_config.aggregate_patches_strategy}_agg_strategy_"
            f"{self.inference_config.num_sampled_random_boxes_per_image}_boxes_"
            f"sigma_{self.inference_config.gaussian_filter_blur_sigma}"
            f"_per_image_statistic_{self.inference_config.image_statistic_for_classification}.csv"
        )

        self.probability_distribution_plot_path = (
            f"{self.metrics_folder}/probability_distribution_plot_{self.inference_config.distance}_distance_"
            f"{self.inference_config.aggregate_patches_strategy}_agg_strategy_"
            f"{self.inference_config.num_sampled_random_boxes_per_image}_boxes_"
            f"sigma_{self.inference_config.gaussian_filter_blur_sigma}"
            f"_per_image_statistic_{self.inference_config.image_statistic_for_classification}.png"
        )

        self.roc_plot_path = (
            f"{self.metrics_folder}/roc_plot_{self.inference_config.distance}_distance_"
            f"{self.inference_config.aggregate_patches_strategy}_agg_strategy_"
            f"{self.inference_config.num_sampled_random_boxes_per_image}_boxes_"
            f"sigma_{self.inference_config.gaussian_filter_blur_sigma}"
            f"_per_image_statistic_{self.inference_config.image_statistic_for_classification}.png"
        )

    def run(self):
        anomaly_maps_paths = [
            file_path for file_path in 
            glob.glob(os.path.join(self.anomaly_maps_arrays_path, "*.npy"))
            if "_anomaly_map_" in file_path
        ]

        segmentation_paths = [
            anomaly_map_path.replace("_anomaly_map_", "_segmentation_mask_") 
            for anomaly_map_path in anomaly_maps_paths]

        roc_auc_metrics = compute_roc_auc_metrics(
            image_statistic_for_classification=self.inference_config.image_statistic_for_classification, 
            anomaly_maps_paths=anomaly_maps_paths, 
            segmentation_paths=segmentation_paths,
            plot_probability_distribution_path=self.probability_distribution_plot_path,
            plot_roc_path=self.roc_plot_path
        )
        print('Image-level ROC-AUC: %.5f' % (roc_auc_metrics["image_roc_auc"]))
        print('Pixel-level ROC-AUC: %.5f' % (roc_auc_metrics["pixel_roc_auc"]))

        metrics_df = pd.DataFrame(
            {
                "DATASET_NAME": [self.inference_config.dataset_name],
                "SPLIT": [self.inference_config.split],
                "AGGREGATE_PATCHES_STRATEGY": [self.inference_config.aggregate_patches_strategy],
                "NUM_TARGET_BOXES_PER_IMAGE": [self.inference_config.num_sampled_random_boxes_per_image],
                "GAUSSIAN_FILTER_BLUR_SIGMA": [self.inference_config.gaussian_filter_blur_sigma],
                "PER_IMAGE_STATISTIC": [self.inference_config.image_statistic_for_classification],
                "Image ROCAUC": [roc_auc_metrics["image_roc_auc"]],
                "Pixel ROCAUC": [roc_auc_metrics["pixel_roc_auc"]],
                "DISTANCE": [self.inference_config.distance],
                "EXPERIMENT_NAME": [self.inference_config.experiment_name],
            }
        ) 
        metrics_df.to_csv(self.metrics_path, index=False)
    

def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config_path", type=str, help="Path to the inference config used for generating the anomaly maps.")

    args = parser.parse_args()
    metrics_computer = MetricsComputer(inference_config_path=args.inference_config_path)
    metrics_computer.run()

if __name__ == "__main__":
    main()
