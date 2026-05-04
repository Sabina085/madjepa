import subprocess
import glob
import argparse


def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Training pipeline - I-Jepa training + distilled model training leveraging the target encoder")
    parser.add_argument(
        "--train_ijepa_config_pattern_path", type=str, help="Pattern for the train I-Jepa configuration files (if running the pipeline on multiple configurations)"
    )
    args = parser.parse_args()

    ijepa_train_configs = sorted(glob.glob(args.train_ijepa_config_pattern_path))

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


if __name__ == "__main__":
    main()
