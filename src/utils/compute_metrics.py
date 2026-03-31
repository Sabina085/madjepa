import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

sns.set_theme()


def compute_roc_auc_metrics(
        image_statistic_for_classification,
        anomaly_maps_paths, 
        segmentation_paths,
        plot_probability_distribution_path,
        plot_roc_path):
    anomaly_maps_list = []
    gt_label_list = []
    gt_segmentations_list = []

    image_name_list = []
    image_scores = []

    for anomaly_map_path, segmentation_path in zip(anomaly_maps_paths, segmentation_paths):
        anomaly_map = np.load(anomaly_map_path)

    # for computing an image level ROCAUC
    # compute an anomaly score at image level
        if image_statistic_for_classification == "mean":
            image_scores.append(np.mean(anomaly_map))
        elif image_statistic_for_classification == "max":
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
    image_roc_auc = roc_auc_score(y_true=gt_label_list, y_score=image_scores)

    display = RocCurveDisplay.from_predictions(y_true=gt_label_list, y_score=image_scores)
    display.plot()
    plt.savefig(plot_roc_path)

    predictions_df = pd.DataFrame({"prediction": image_scores, "label": gt_label_list})
    sns.displot(predictions_df, x="prediction", hue="label", kind="hist")
    plt.savefig(plot_probability_distribution_path)
    plt.close()

    # Calculate per-pixel level ROCAUC
    y_masks_gt = np.concatenate(gt_segmentations_list).astype('int')
    y_amaps_pred = np.concatenate(anomaly_maps_list).astype('float32')

    fpr_pixel, tpr_pixel, thresholds_pixel = roc_curve(
    y_true=y_masks_gt.flatten(), y_score=y_amaps_pred.flatten())

    pixel_roc_auc = roc_auc_score(
    y_true=y_masks_gt.flatten(), y_score=y_amaps_pred.flatten())

    return {
        "image_roc_auc": image_roc_auc,
        "pixel_roc_auc": pixel_roc_auc
    }
