import subprocess
import glob

inference_configs = [glob.glob("configs/inference_anomaly_detection/RESC.yaml")]

print("Number of inference configs: ", len(inference_configs))
train_distillation_config_path = "experiments_RESC/experiment_default_hyparams_early_stopping/checkpoints_distilled_model/430b5f5c399a4fc2/config.json"

for inference_config_path in inference_configs:
    
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
