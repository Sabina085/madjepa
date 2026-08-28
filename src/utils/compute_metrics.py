import numpy as np
import pandas as pd
import json
import cv2
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from PIL import Image
from src.utils.anomaly_maps import cvt2heatmap

sns.set_theme()


def compute_roc_auc_metrics(
        image_statistic_for_classification,
        anomaly_maps_paths, 
        anomaly_maps_distillation,
        classifier_results_paths,
        segmentation_paths,
        plot_probability_distribution_path,
        plot_roc_path,
        weight_distillation,
        weight_classification
    ):
    anomaly_maps_list = []
    gt_label_list = []
    gt_segmentations_list = []

    image_name_list = []
    image_scores = []

    for anomaly_map_path, anomaly_map_distillation, classifier_results_path, segmentation_path in zip(anomaly_maps_paths, anomaly_maps_distillation, classifier_results_paths, segmentation_paths):
        anomaly_map = np.load(anomaly_map_path)
        anomaly_map_distillation = np.load(anomaly_map_distillation)
        with open(classifier_results_path, "r") as f:
            classification_score = float(json.load(f)["score"])

        anomaly_map = anomaly_map * (1 - weight_distillation) + anomaly_map_distillation * weight_distillation

        # Saving the aggregated anomaly map (I-Jepa + distillation) and the heatmap
        anomaly_map_image_path = anomaly_map_path.replace("_arrays_", "_images_").replace(".npy", ".png")
        agg_anomaly_map_path  = anomaly_map_image_path.replace("_anomaly_map_", "_aggregated_anomaly_map_")
        agg_anomaly_map_path = agg_anomaly_map_path.split(".png")[0] + "_" + str(weight_distillation) + ".png"
        
        Image.fromarray((255 * anomaly_map) \
            .astype("uint8")) \
            .convert("L") \
            .save(agg_anomaly_map_path)
        
        # Convert the anomaly map into a heatmap and superimpose it on the initial image
        heatmap_agg_anomaly_map = cvt2heatmap((anomaly_map * 255).astype(np.uint8))
        heatmap_agg_anomaly_map_path = agg_anomaly_map_path.replace("_aggregated_anomaly_map_", "_").split(".png")[0] + "_aggregated_heatmap_" + str(weight_distillation) + ".png"

        initial_image = cv2.imread(anomaly_map_image_path.replace("_anomaly_map_", "_").split(".png")[0] + "_original.png")
        initial_image = (initial_image * 255).astype(np.uint8)

        heatmap_agg_anomaly_map = cv2.addWeighted(heatmap_agg_anomaly_map, 0.4, initial_image, (1 - 0.4), 0)
        cv2.imwrite(heatmap_agg_anomaly_map_path, heatmap_agg_anomaly_map)

        # for computing an image level ROCAUC
        # compute an anomaly score at image level
        if image_statistic_for_classification == "mean":
            anomaly_map_score = np.mean(anomaly_map)
        elif image_statistic_for_classification == "max":
            anomaly_map_score = np.max(anomaly_map)
            
        image_score = anomaly_map_score * (1 - weight_classification) + classification_score * weight_classification
        image_scores.append(image_score)

        gt_label_list.append(int(anomaly_map_path[-5]))

        if segmentation_path:
            # For computing pixel level ROCAUC
            anomaly_maps_list.append(anomaly_map)
            gt_segmentations_list.append(np.load(segmentation_path)) 

            image_name_list.append(anomaly_map_path.split("/")[-1])
    
    print("Number of anomalous samples: ", sum(gt_label_list))
    print("Total number of samples: ", len(gt_label_list))

    # Calculate image-level ROCAUC score
    image_roc_auc = roc_auc_score(y_true=gt_label_list, y_score=image_scores)

    display = RocCurveDisplay.from_predictions(y_true=gt_label_list, y_score=image_scores)
    display.plot()
    plt.savefig(plot_roc_path)

    predictions_df = pd.DataFrame({"anomaly_score": image_scores, "label": gt_label_list})
    sns.displot(predictions_df, x="anomaly_score", hue="label", kind="hist", bins=50)
    plt.savefig(plot_probability_distribution_path)
    plt.close()

    if segmentation_path:
        # Calculate per-pixel level ROCAUC
        y_masks_gt = np.concatenate(gt_segmentations_list, dtype=np.uint8).ravel()
        y_amaps_pred = np.concatenate(anomaly_maps_list, dtype=np.float32).ravel()

        del anomaly_maps_list, gt_segmentations_list

        pixel_roc_auc = roc_auc_score(
            y_true=y_masks_gt, y_score=y_amaps_pred)
    else:
         # the current dataset does not have segmentation masks available
         pixel_roc_auc = None

    return {
        "image_roc_auc": image_roc_auc,
        "pixel_roc_auc": pixel_roc_auc
    }
