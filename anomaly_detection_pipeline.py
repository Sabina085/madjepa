import subprocess
import glob
import argparse
import yaml
from instance_specific_config import ROOT_DIRECTORY_EXPERIMENTS_PATH

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

        train_distillation_config_path = glob.glob(f"{ROOT_DIRECTORY_EXPERIMENTS_PATH}/experiments_{dataset_name}/{experiment_name}/checkpoints_distilled_model/*/config.yaml")[0]    
        print("Train distillation config path: ", train_distillation_config_path)

        scripts = [
            ["python3", "-m", "src.inference.generate_predictions_random_boxes", f"--inference_config_path={inference_config_path}"],
            ["python3", "-m", "src.inference.generate_predictions_distilled_model", f"--inference_config_path={inference_config_path}", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.build_anomaly_maps", f"--inference_config_path={inference_config_path}"],
            ["python3", "-m", "src.evaluation.build_anomaly_maps", f"--inference_config_path={inference_config_path}", "--use_distillation_model", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.5", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=1.0", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.05", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.1", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.15", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.2", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.25", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.3", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.35", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.4", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.45", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.75", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.9", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.95", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.85", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.91", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.92", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.93", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.94", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.89", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.88", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.87", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.86", f"--train_distilled_model_config_path={train_distillation_config_path}"],
        ]

        for cmd in scripts:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd)

            if result.returncode != 0:
                print("Error occurred, stopping execution.")
                break

if __name__ == "__main__":
    main()
