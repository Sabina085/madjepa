import subprocess
import glob
from pathlib import Path
import os
import argparse
import yaml
from instance_specific_config import ROOT_DIRECTORY_EXPERIMENTS_PATH

DISTILLATION_MODEL_HASH = "*" # default value - retrieve the only checkpoint available for the experiment indicated in inference config

def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Pattern for the inference configuration files")
    parser.add_argument(
        "--inference_config_pattern_path", type=str, help="Pattern for the inference configuration files"
    )
    args = parser.parse_args()
    inference_configs = sorted(glob.glob(args.inference_config_pattern_path))
    print("Number of inference configs: ", len(inference_configs))

    for inference_config_path in inference_configs:
        print("Inference config: ", inference_config_path)
        with open(inference_config_path, "r") as f:
            inference_config = yaml.safe_load(f)

        experiment_name = inference_config["data"]["experiment_name"]
        dataset_name = inference_config["data"]["dataset_name"]

        # Distilled model
        train_distillation_config_path = glob.glob(
            f"{ROOT_DIRECTORY_EXPERIMENTS_PATH}/experiments_{dataset_name}/{experiment_name}/checkpoints_distilled_model/{DISTILLATION_MODEL_HASH}/config.yaml")[0]    
        print("Train distillation config path: ", train_distillation_config_path)

        # Classifier model
        # Select classifier checkpoint with best validation AUC (encoded in filename)
        classifier_candidates = sorted(
            glob.glob(f"{ROOT_DIRECTORY_EXPERIMENTS_PATH}/experiments_{dataset_name}/{experiment_name}/checkpoints_classifier/*/*.pth")
        )
        best_classifier_path = classifier_candidates[0]
        best_classifier_auc = float(Path(best_classifier_path).stem.split("-")[-1])
        for path in classifier_candidates[1:]:
            auc = float(Path(path).stem.split("-")[-1])
            if auc > best_classifier_auc:
                best_classifier_path = path
                best_classifier_auc = auc

        train_classifier_ckpt_path = best_classifier_path
        print("train_classifier_ckpt_path: ", train_classifier_ckpt_path)
        train_classifier_config_path = os.path.join(Path(train_classifier_ckpt_path).parent, "config.yaml")
        print("train_classifier_config_path: ", train_classifier_config_path)

        scripts = [
            # generate predictions
            ["python3", "-m", "src.inference.generate_predictions_random_boxes", f"--inference_config_path={inference_config_path}"],
            ["python3", "-m", "src.inference.generate_predictions_distilled_model", f"--inference_config_path={inference_config_path}", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.inference.generate_predictions_classifier_model", f"--inference_config_path={inference_config_path}", f"--train_classification_model_config_path={train_classifier_config_path}"],
            
            # generate anomaly maps
            ["python3", "-m", "src.evaluation.build_anomaly_maps", f"--inference_config_path={inference_config_path}"],
            ["python3", "-m", "src.evaluation.build_anomaly_maps", f"--inference_config_path={inference_config_path}", "--use_distillation_model", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            
            # Compute the anomaly detection results

            # Vary the weight_distillation, weight_classification=0.0
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.0","--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.5","--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=1.0","--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.05","--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.1", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.15", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.2", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.25", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.3", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.35", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.4", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.45", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.6", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.7", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.75", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.8", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.85", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],

            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.9", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.95", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.91", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.92", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.93", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.94", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.89", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.88", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.87", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.86", "--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
        
            # Vary the weight_classificaton, weight_distillation is fixed
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.5", "--weight_classification=0.1", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.5", "--weight_classification=0.2", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.5", "--weight_classification=0.3", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.5", "--weight_classification=0.4", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.5", "--weight_classification=0.5", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.5", "--weight_classification=0.6", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.5", "--weight_classification=0.7", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.5", "--weight_classification=0.8", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.5", "--weight_classification=0.9", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.5", "--weight_classification=1.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"]
        ]

        for cmd in scripts:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd)

            if result.returncode != 0:
                print("Error occurred, stopping execution.")
                break

if __name__ == "__main__":
    main()
