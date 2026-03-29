import os
import glob
import pandas as pd
from src.utils.compute_metrics import compute_roc_auc_metrics


subpath_list = os.getcwd().split(os.sep)
root_directory_path = os.path.join(*subpath_list[:-1])
current_dir = subpath_list[-1]

###  To modify these configuration values
DATASET_NAME = "BraTS2021_slice"
SPLIT = "valid"
AGGREGATE_PATCHES_STRATEGY = "mean"
NUM_TARGET_BOXES_PER_IMAGE = 30
GAUSSIAN_FILTER_BLUR_SIGMA = 4
DISTANCE = "l2"

# For computing the image-level metrics
PER_IMAGE_STATISTIC = "max"
###

EXPERIMENT_FOLDER_NAME = "enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3_cropped_random_crop_training"

EXPERIMENT_FOLDER_PATH = (
    f"/{root_directory_path}/{current_dir}/"
    f"experiments_{DATASET_NAME}/{EXPERIMENT_FOLDER_NAME}/results_{SPLIT}"
)

ANOMALY_MAPS_ARRAYS_PATH = (
    f"{EXPERIMENT_FOLDER_PATH}/"
    f"anomaly_maps_arrays_{DISTANCE}_distance_{AGGREGATE_PATCHES_STRATEGY}_agg_strategy_"
    f"{NUM_TARGET_BOXES_PER_IMAGE}_boxes_sigma_{GAUSSIAN_FILTER_BLUR_SIGMA}"
)

METRICS_FOLDER = f"{EXPERIMENT_FOLDER_PATH}/Metrics"
os.makedirs(METRICS_FOLDER, exist_ok=True)

METRICS_PATH = (
    f"{METRICS_FOLDER}/metrics_{DISTANCE}_distance_{AGGREGATE_PATCHES_STRATEGY}_agg_strategy_"
    f"{NUM_TARGET_BOXES_PER_IMAGE}_boxes_sigma_{GAUSSIAN_FILTER_BLUR_SIGMA}_per_image_statistic_{PER_IMAGE_STATISTIC}.csv"
)

if __name__ == "__main__":
    anomaly_maps_paths = [
        file_path for file_path in 
        glob.glob(os.path.join(ANOMALY_MAPS_ARRAYS_PATH, "*.npy")) if "_anomaly_map_" in file_path]

    segmentation_paths = [
        anomaly_map_path.replace("_anomaly_map_", "_segmentation_mask_") 
        for anomaly_map_path in anomaly_maps_paths]


    roc_auc_metrics = compute_roc_auc_metrics(PER_IMAGE_STATISTIC, anomaly_maps_paths, segmentation_paths)
    print('Image-level ROC-AUC: %.5f' % (roc_auc_metrics["image_roc_auc"]))
    print('Pixel-level ROC-AUC: %.5f' % (roc_auc_metrics["pixel_roc_auc"]))

    metrics_df = pd.DataFrame(
        {
            "DATASET_NAME": [DATASET_NAME],
            "SPLIT": [SPLIT],
            "AGGREGATE_PATCHES_STRATEGY": [AGGREGATE_PATCHES_STRATEGY],
            "NUM_TARGET_BOXES_PER_IMAGE": [NUM_TARGET_BOXES_PER_IMAGE],
            "GAUSSIAN_FILTER_BLUR_SIGMA": [GAUSSIAN_FILTER_BLUR_SIGMA],
            "PER_IMAGE_STATISTIC": [PER_IMAGE_STATISTIC],
            "Image ROCAUC": [roc_auc_metrics["image_roc_auc"]],
            "Pixel ROCAUC": [roc_auc_metrics["pixel_roc_auc"]],
            "DISTANCE": [DISTANCE],
            "EXPERIMENT_NAME": [EXPERIMENT_FOLDER_NAME],
        }
    ) 
    metrics_df.to_csv(METRICS_PATH, index=False)
