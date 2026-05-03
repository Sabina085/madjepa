import subprocess
import glob

ijepa_train_configs = sorted(glob.glob("configs/train_ijepa/hist_DIY_experiment_wd_4_no_normalization.yaml"))

print("Number of ijepa training configs: ", len(ijepa_train_configs))

train_distillation_configs = [ijepa_train_config.replace("train_ijepa", "train_distillation") for ijepa_train_config in ijepa_train_configs]

for (ijepa_train_config, distillation_train_config) in zip(ijepa_train_configs, train_distillation_configs):
    scripts = [
        ["python3", "main.py", f"--fname={ijepa_train_config}"],
        ["python3", "-m", "src.train_distilled_model", f"--train_distillation_config_path={distillation_train_config}"]
    ]

    for cmd in scripts:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print("Error occurred, stopping execution.")
            break
