import subprocess
import glob
from pathlib import Path
import os
import argparse
import yaml
import time
from instance_specific_config import ROOT_DIRECTORY_EXPERIMENTS_PATH


def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Pattern for the inference configuration files")
    parser.add_argument(
        "--inference_config_pattern_path", type=str, help="Pattern for the inference configuration files"
    )

    parser.add_argument(
        "--distillation_model_hash", type=str, default="*", help="Distilled model hash. Default = * (the only checkpoint available for the experiment indicated in the inference config)"
    )

    parser.add_argument(
        "--weight_distillation_test", type=float, default=0.0, help="Weight of the distilled model to use on the test set (best value identified on the validation set). Required only for computing the results on the test set."
    )

    parser.add_argument(
        "--weight_classification_test", type=float, default=0.0, help="Weight of the classifier model to use on the test set (best value identified on the validation set). Required only for computing the results on the test set."
    )

    args = parser.parse_args()
    distillation_model_hash = str(args.distillation_model_hash)
    weight_distillation_test = float(args.weight_distillation_test)
    weight_classification_test = float(args.weight_classification_test)
    inference_configs = sorted(glob.glob(args.inference_config_pattern_path))
    print("Number of inference configs: ", len(inference_configs))
    # time.sleep(1800)

    for inference_config_path in inference_configs:
        print("Inference config: ", inference_config_path)
        with open(inference_config_path, "r") as f:
            inference_config = yaml.safe_load(f)

        experiment_name = inference_config["data"]["experiment_name"]
        dataset_name = inference_config["data"]["dataset_name"]

        # Distilled model
        train_distillation_config_path = glob.glob(
            f"{ROOT_DIRECTORY_EXPERIMENTS_PATH}/experiments_{dataset_name}/{experiment_name}/checkpoints_distilled_model/{distillation_model_hash}/config.yaml")[0]    
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
            # generate prsedictions
            ["python3", "-m", "src.inference.generate_predictions_random_boxes", f"--inference_config_path={inference_config_path}"],
            ["python3", "-m", "src.inference.generate_predictions_distilled_model", f"--inference_config_path={inference_config_path}", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.inference.generate_predictions_classifier_model", f"--inference_config_path={inference_config_path}", f"--train_classification_model_config_path={train_classifier_config_path}"],
            
            # generate anomaly maps
            ["python3", "-m", "src.evaluation.build_anomaly_maps", f"--inference_config_path={inference_config_path}"],
            ["python3", "-m", "src.evaluation.build_anomaly_maps", f"--inference_config_path={inference_config_path}", "--use_distillation_model", f"--train_distilled_model_config_path={train_distillation_config_path}"],

            # ijepa model only
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.0","--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # distilled model only
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=1.0","--weight_classification=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
            # pseudo-anomaly classifier model only
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.0","--weight_classification=1.0", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"],
        ]

        # Compute the anomaly detection results
        if inference_config["data"]["split"] == "valid":
            # Validation set - vary both the distilled model weight and the classifier model weight
            weights_distillation = [round(0.05 * x, 2) for x in range(0, 20)]
            weights_classification = [round(0.05 * x, 2) for x in range(0, 20)]

            for weight_distillation in weights_distillation:
                for weight_classification in weights_classification:
                    scripts.append(
                        ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", f"--weight_distillation={weight_distillation}", f"--weight_classification={weight_classification}", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"]
                    )
        else:
            # Test set - compute performance using the best weight_distillation_test and weight_classification_test previously identified on the validation set
            scripts.append(
                ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", f"--weight_distillation={weight_distillation_test}", f"--weight_classification={weight_classification_test}", f"--train_distilled_model_config_path={train_distillation_config_path}", f"--train_classifier_config_path={train_classifier_config_path}"]
            )
            
        for cmd in scripts:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd)

            if result.returncode != 0:
                print("Error occurred, stopping execution.")
                break

if __name__ == "__main__":
    main()
