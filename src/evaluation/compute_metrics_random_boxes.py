import os
import numpy as np
import glob
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt 

sns.set_theme()

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
# COUNT_ANOMALY_PIXELS_IN_CONTEXT = 10

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
    # f"anomaly_pixels_in_context_{COUNT_ANOMALY_PIXELS_IN_CONTEXT}"
)

METRICS_FOLDER = f"{EXPERIMENT_FOLDER_PATH}/Metrics"
os.makedirs(METRICS_FOLDER, exist_ok=True)

METRICS_PATH = (
    f"{METRICS_FOLDER}/metrics_{DISTANCE}_distance_{AGGREGATE_PATCHES_STRATEGY}_agg_strategy_"
    f"{NUM_TARGET_BOXES_PER_IMAGE}_boxes_sigma_{GAUSSIAN_FILTER_BLUR_SIGMA}_per_image_statistic_{PER_IMAGE_STATISTIC}.csv"
    # f"anomaly_pixels_in_context_{COUNT_ANOMALY_PIXELS_IN_CONTEXT}.csv"
)

anomaly_maps_paths = [
    file_path for file_path in 
    glob.glob(os.path.join(ANOMALY_MAPS_ARRAYS_PATH, "*.npy")) if "_anomaly_map_" in file_path]

segmentation_paths = [
    anomaly_map_path.replace("_anomaly_map_", "_segmentation_mask_") 
    for anomaly_map_path in anomaly_maps_paths]

anomaly_maps_list = []
gt_label_list = []
gt_segmentations_list = []

image_name_list = []
image_scores = []

for anomaly_map_path, segmentation_path in zip(anomaly_maps_paths, segmentation_paths):
    anomaly_map = np.load(anomaly_map_path)

    # for computing an image level ROCAUC
    # compute an anomaly score at image level
    if PER_IMAGE_STATISTIC == "mean":
        image_scores.append(np.mean(anomaly_map))
    elif PER_IMAGE_STATISTIC == "max":
        image_scores.append(np.max(anomaly_map))
    gt_label_list.append(int(anomaly_map_path[-5]))

    # For computing pixel level ROCAUC
    anomaly_maps_list.append(anomaly_map)
    gt_segmentations_list.append(np.load(segmentation_path)) 

    image_name_list.append(anomaly_map_path.split("/")[-1])
    
print("Number of anomalous samples: ", sum(gt_label_list))
print("Total number of samples: ", len(gt_label_list))

# Calculate image-level ROC AUC score
fprs, tprs, thresholds = roc_curve(y_true=gt_label_list, y_score=image_scores)
for fpr, tpr, threshold in zip(fprs, tprs, thresholds):
    print(f"threshold: {threshold} -> fpr: {fpr}, tpr: {tpr}")
img_roc_auc = roc_auc_score(y_true=gt_label_list, y_score=image_scores)
print('image ROCAUC: %.5f' % (img_roc_auc))

display = RocCurveDisplay.from_predictions(y_true=gt_label_list, y_score=image_scores)
display.plot()
plt.savefig("Image_roc_curve.png")

# for pred, image_name, label in zip(image_scores, image_name_list, gt_label_list):
#     print("~~~" + str(pred) + ", " + image_name + ", " + str(label))

predictions_df = pd.DataFrame({"prediction": image_scores, "label": gt_label_list})
sns.displot(predictions_df, x="prediction", hue="label", kind="hist")
plt.savefig("my_plot.png")
plt.close()

# Calculate per-pixel level ROCAUC
y_masks_gt = np.concatenate(gt_segmentations_list).astype('int')
y_amaps_pred = np.concatenate(anomaly_maps_list).astype('float32')

fpr_pixel, tpr_pixel, thresholds_pixel = roc_curve(
    y_true=y_masks_gt.flatten(), y_score=y_amaps_pred.flatten())

per_pixel_rocauc = roc_auc_score(
    y_true=y_masks_gt.flatten(), y_score=y_amaps_pred.flatten()
)
print('pixel ROCAUC: %.5f' % (per_pixel_rocauc))

metrics_df = pd.DataFrame(
    {
        "DATASET_NAME": [DATASET_NAME],
        "SPLIT": [SPLIT],
        "AGGREGATE_PATCHES_STRATEGY": [AGGREGATE_PATCHES_STRATEGY],
        "NUM_TARGET_BOXES_PER_IMAGE": [NUM_TARGET_BOXES_PER_IMAGE],
        "GAUSSIAN_FILTER_BLUR_SIGMA": [GAUSSIAN_FILTER_BLUR_SIGMA],
        "PER_IMAGE_STATISTIC": [PER_IMAGE_STATISTIC],
        "Image ROCAUC": [img_roc_auc],
        "Pixel ROCAUC": [per_pixel_rocauc],
        "DISTANCE": [DISTANCE],
        "EXPERIMENT_NAME": [EXPERIMENT_FOLDER_NAME],
        # "COUNT_ANOMALY_PIXELS": [COUNT_ANOMALY_PIXELS_IN_CONTEXT]
    }
) 

metrics_df.to_csv(METRICS_PATH, index=False)
