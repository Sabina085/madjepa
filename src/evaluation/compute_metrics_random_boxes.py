import os
import argparse
import glob
import pandas as pd
from pathlib import Path
from src.utils.configs import *
from src.utils.compute_metrics import compute_roc_auc_metrics
from instance_specific_config import ROOT_DIRECTORY_EXPERIMENTS_PATH

class MetricsComputer:
    def __init__(self, 
            inference_config_path: str, 
            train_distilled_model_config_path: str, 
            weight_distillation: float=0.0,
            weight_classification: float=0.0
        ):
        self.inference_config = read_inference_anomaly_detection_config(
            config_path=inference_config_path)

        self.weight_distillation = weight_distillation
        self.weight_classification = weight_classification

        self.experiment_folder_path = (
            f"{ROOT_DIRECTORY_EXPERIMENTS_PATH}/"
            f"experiments_{self.inference_config.dataset_name}/"
            f"{self.inference_config.experiment_name}/results_{self.inference_config.split}"
        )
        
        # I-Jepa anomaly maps
        self.anomaly_maps_arrays_path = (
            f"{self.experiment_folder_path}/"
            f"anomaly_maps_arrays_{self.inference_config.distance}_distance_"
            f"{self.inference_config.aggregate_patches_strategy}_agg_strategy_"
            f"{self.inference_config.num_sampled_random_boxes_per_image}_boxes_"
            f"sigma_{self.inference_config.gaussian_filter_blur_sigma_ijepa}"
        )

        # Distilled model anomaly maps
        self.distilled_model_hash_folder = train_distilled_model_config_path.split("/")[-2]
        self.anomaly_maps_distillation_arrays_path = (
            f"{self.experiment_folder_path}/"
            f"anomaly_maps_arrays_distillation_{self.inference_config.distance}_distance_"
            f"sigma_{self.inference_config.gaussian_filter_blur_sigma_distillation}_checkpoint_{self.distilled_model_hash_folder}"
        )

        # Pseudo-anomaly classifier results
        self.classifier_hash_folder = "0aea96f5663b0527" # Path(self.classifier_ckpt_path).parent.name
        self.classifier_results_folder = os.path.join(
            self.experiment_folder_path,
            f"image_classifier_results_checkpoint_{self.classifier_hash_folder}"
        )

        self.metrics_folder = f"{self.experiment_folder_path}/Metrics"
        os.makedirs(self.metrics_folder, exist_ok=True)

        self.metrics_path = (
            f"{self.metrics_folder}/metrics_{self.inference_config.distance}_distance_"
            f"{self.inference_config.aggregate_patches_strategy}_agg_strategy_"
            f"{self.inference_config.num_sampled_random_boxes_per_image}_boxes_"
            f"sigma_ijepa_{self.inference_config.gaussian_filter_blur_sigma_ijepa}_"
            f"sigma_distillation_{self.inference_config.gaussian_filter_blur_sigma_distillation}_"
            f"_per_image_statistic_{self.inference_config.image_statistic_for_classification}"
            f"_checkpoint_{self.distilled_model_hash_folder}"
            f"_distillation_weight_{self.weight_distillation}"
            f"_classification_weight_{self.weight_classification}.csv"
        )

        self.probability_distribution_plot_path = (
            f"{self.metrics_folder}/probability_distribution_plot_{self.inference_config.distance}_distance_"
            f"{self.inference_config.aggregate_patches_strategy}_agg_strategy_"
            f"{self.inference_config.num_sampled_random_boxes_per_image}_boxes_"
            f"sigma_ijepa_{self.inference_config.gaussian_filter_blur_sigma_ijepa}_"
            f"sigma_distillation_{self.inference_config.gaussian_filter_blur_sigma_distillation}_"
            f"_per_image_statistic_{self.inference_config.image_statistic_for_classification}"
            f"_checkpoint_{self.distilled_model_hash_folder}"
            f"_distillation_weight_{self.weight_distillation}"
            f"_classification_weight_{self.weight_classification}.png"
        )

        self.roc_plot_path = (
            f"{self.metrics_folder}/roc_plot_{self.inference_config.distance}_distance_"
            f"{self.inference_config.aggregate_patches_strategy}_agg_strategy_"
            f"{self.inference_config.num_sampled_random_boxes_per_image}_boxes_"
            f"sigma_ijepa_{self.inference_config.gaussian_filter_blur_sigma_ijepa}"
            f"sigma_distillation_{self.inference_config.gaussian_filter_blur_sigma_distillation}"
            f"_per_image_statistic_{self.inference_config.image_statistic_for_classification}"
            f"_checkpoint_{self.distilled_model_hash_folder}"
            f"_distillation_weight_{self.weight_distillation}"
            f"_classification_weight_{self.weight_classification}.png"
        )

    def run(self):
        anomaly_maps_paths = [
            file_path for file_path in 
            glob.glob(os.path.join(self.anomaly_maps_arrays_path, "*.npy"))
            if "_anomaly_map_" in file_path
        ]

        anomaly_maps_distillation_paths = []
        for anomaly_maps_path in anomaly_maps_paths:
            anomaly_maps_distillation_paths.append(
                os.path.join(
                    self.anomaly_maps_distillation_arrays_path,
                    Path(anomaly_maps_path).name
                )
        )
        classifier_results_paths = []
        for anomaly_maps_path in anomaly_maps_paths:
            classifier_file_name = Path(anomaly_maps_path).stem.replace("anomaly_map", "score") + ".json"
            classifer_result_path = os.path.join(
                    self.classifier_results_folder,
                    classifier_file_name
            )
            classifier_results_paths.append(classifer_result_path)


        if self.inference_config.dataset_name in ["BraTS2021_slice", "hist_DIY", "RESC"]:
            segmentation_paths = [
                anomaly_map_path.replace("_anomaly_map_", "_segmentation_mask_") 
                for anomaly_map_path in anomaly_maps_paths]
        else:
            segmentation_paths = [None] * len(anomaly_maps_paths)

        roc_auc_metrics = compute_roc_auc_metrics(
            image_statistic_for_classification=self.inference_config.image_statistic_for_classification, 
            anomaly_maps_paths=anomaly_maps_paths, 
            anomaly_maps_distillation=anomaly_maps_distillation_paths,
            classifier_results_paths=classifier_results_paths,
            segmentation_paths=segmentation_paths,
            plot_probability_distribution_path=self.probability_distribution_plot_path,
            plot_roc_path=self.roc_plot_path,
            weight_distillation=self.weight_distillation,
            weight_classification=self.weight_classification
        )
        print('Image-level ROC-AUC: %.5f' % (roc_auc_metrics["image_roc_auc"]))
        if self.inference_config.dataset_name in ["BraTS2021_slice", "hist_DIY", "RESC"]:
            print('Pixel-level ROC-AUC: %.5f' % (roc_auc_metrics["pixel_roc_auc"]))

        metrics_df = pd.DataFrame(
            {
                "DATASET_NAME": [self.inference_config.dataset_name],
                "SPLIT": [self.inference_config.split],
                "AGGREGATE_PATCHES_STRATEGY": [self.inference_config.aggregate_patches_strategy],
                "NUM_TARGET_BOXES_PER_IMAGE": [self.inference_config.num_sampled_random_boxes_per_image],
                "GAUSSIAN_FILTER_BLUR_SIGMA_IJEPA": [self.inference_config.gaussian_filter_blur_sigma_ijepa],
                "GAUSSIAN_FILTER_BLUR_SIGMA_DISTILLATION": [self.inference_config.gaussian_filter_blur_sigma_distillation],
                "PER_IMAGE_STATISTIC": [self.inference_config.image_statistic_for_classification],
                "Image ROCAUC": [roc_auc_metrics["image_roc_auc"]],
                "Pixel ROCAUC": [roc_auc_metrics["pixel_roc_auc"]],
                "DISTANCE": [self.inference_config.distance],
                "WEIGHT_DISTILLATION": [self.weight_distillation],
                "WEIGHT_CLASSIFICATION": [self.weight_classification],
                "EXPERIMENT_NAME": [self.inference_config.experiment_name],
                "DISTILLED_MODEL_HASH": [self.distilled_model_hash_folder]
            }
        ) 
        metrics_df.to_csv(self.metrics_path, index=False)
    

def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config_path", type=str, help="Path to the inference config used for generating the anomaly maps.")
    parser.add_argument("--train_distilled_model_config_path", type=str, help="Path to the training config of the distilled model which will be used for computing the results")
    parser.add_argument("--weight_distillation", type=float, default=0.0, help="Weight of the distillation anomaly map when computing the metrics.")
    parser.add_argument("--weight_classification", type=float, default=0.0, help="Weight of the pseudo-anomaly classification score when computing the metrics.")

    args = parser.parse_args()
    metrics_computer = MetricsComputer(
        inference_config_path=args.inference_config_path,
        train_distilled_model_config_path=args.train_distilled_model_config_path,
        weight_distillation=float(args.weight_distillation),
        weight_classification=float(args.weight_classification)
    )
    metrics_computer.run()

if __name__ == "__main__":
    main()
