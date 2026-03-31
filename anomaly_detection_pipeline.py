import subprocess

scripts = [
    ["python3", "-m", "src.inference.generate_predictions_random_boxes", "--inference_config_path=configs/inference_anomaly_detection/BraTS2021_slice.yaml"],
    ["python3", "-m", "src.evaluation.build_anomaly_maps", "--inference_config_path=configs/inference_anomaly_detection/BraTS2021_slice.yaml"],
    ["python3", "-m", "src.evaluation.compute_metrics_random_boxes", "--inference_config_path=configs/inference_anomaly_detection/BraTS2021_slice.yaml"]
]

for cmd in scripts:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Error occurred, stopping execution.")
        break
