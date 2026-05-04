import subprocess
import glob
import argparse
import yaml

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

        train_distillation_config_path = glob.glob(f"/media/vhondru/hdd/istoian/ijepa-anomaly-detection/experiments_{dataset_name}/{experiment_name}/checkpoints_distilled_model/*/config.json")[0]    
        print("Train distillation config path: ", train_distillation_config_path)

        scripts = [
            ["python3", "-m", "src.inference.generate_predictions_random_boxes", f"--inference_config_path={inference_config_path}"],
            ["python3", "-m", "src.inference.generate_predictions_distilled_model", f"--inference_config_path={inference_config_path}", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.build_anomaly_maps", f"--inference_config_path={inference_config_path}"],
            ["python3", "-m", "src.evaluation.build_anomaly_maps", f"--inference_config_path={inference_config_path}", "--use_distillation_model", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.0", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=0.5", f"--train_distilled_model_config_path={train_distillation_config_path}"],
            ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", f"--inference_config_path={inference_config_path}", "--weight_distillation=1.0", f"--train_distilled_model_config_path={train_distillation_config_path}"]
        ]

        for cmd in scripts:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd)

            if result.returncode != 0:
                print("Error occurred, stopping execution.")
                break

if __name__ == "__main__":
    main()
