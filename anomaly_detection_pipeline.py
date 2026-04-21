import subprocess

# train_distillation_config_path="experiments_BraTS2021_slice/test_experiment/checkpoints_distilled_model/31151a71e5ad0161/config.json"
# inference_config_path = "configs/inference_anomaly_detection/BraTS2021_slice.yaml"

train_distillation_config_path="experiments_BraTS2021_slice/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3_cropped_random_crop_training/checkpoints_distilled_model/a5bd70b66a814550/config.json"
inference_config_path = "configs/inference_anomaly_detection/BraTS2021_slice.yaml"

# train_distillation_config_path="experiments_hist_DIY/experiment_default_hyparams/experiments_RESC/experiment_default_hyparams/checkpoints_distilled_model/0e261772b74f7484/config.json"
# inference_config_path = "configs/inference_anomaly_detection/hist_DIY.yaml"

# train_distillation_config_path="experiments_RESC/experiment_default_hyparams/checkpoints_distilled_model/92c2d57cfa9eccb0/config.json"
# inference_config_path = "configs/inference_anomaly_detection/RESC.yaml"

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
