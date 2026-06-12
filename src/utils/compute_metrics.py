import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score, RocCurveDisplay

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

    # Calculate image-level ROC AUC score
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
         pixel_roc_auc = None

    return {
        "image_roc_auc": image_roc_auc,
        "pixel_roc_auc": pixel_roc_auc
    }
