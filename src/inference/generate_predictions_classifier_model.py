import torch
import os
import glob
from pathlib import Path
import json
import argparse

from torchvision import transforms
from PIL import Image

from src.utils.configs import (
    read_inference_anomaly_detection_config,
    read_train_ijepa_config,
    read_train_synthetic_anomaly_classifier_config,
)
from src.helper import load_models_from_checkpoint, init_model_dinov3
from src.utils.dataset import *
from src.train_synthetic_anomaly_classifier import HeadClassifier, extract_features

from instance_specific_config import DATASET_PATH, ROOT_DIRECTORY_EXPERIMENTS_PATH

STORAGE_PARENT_FOLDER_PATH = DATASET_PATH


class PredictionsClassifierModelGenerator:
    def __init__(self, inference_config_path: str):
        self.inference_anomaly_detection_config = read_inference_anomaly_detection_config(
            config_path=inference_config_path
        )

        self.experiment_folder_path = (
            f"{ROOT_DIRECTORY_EXPERIMENTS_PATH}/"
            f"experiments_{self.inference_anomaly_detection_config.dataset_name}/"
            f"{self.inference_anomaly_detection_config.experiment_name}"
        )

        self.ijepa_train_config_path = os.path.join(self.experiment_folder_path, "logs/config_run.yaml")
        self.ijepa_train_config = read_train_ijepa_config(self.ijepa_train_config_path)

        ijepa_ckpt_folder_path = os.path.join(self.experiment_folder_path, "checkpoints")
        self.ijepa_ckpt_file_path = os.path.join(
            ijepa_ckpt_folder_path,
            f"{self.inference_anomaly_detection_config.experiment_name}-latest.pth.tar",
        )

        # Select classifier checkpoint with best validation AUC (encoded in filename)
        candidates = sorted(
            glob.glob(os.path.join(self.experiment_folder_path, "checkpoints_classifier/*/*.pth"))
        )
        best_path = candidates[0]
        best_auc = float(Path(best_path).stem.split("-")[-1])
        for path in candidates[1:]:
            auc = float(Path(path).stem.split("-")[-1])
            if auc > best_auc:
                best_path = path
                best_auc = auc

        self.classifier_ckpt_path = best_path
        self.classifier_config_path = os.path.join(Path(best_path).parent, "config.yaml")
        self.classifier_config = read_train_synthetic_anomaly_classifier_config(self.classifier_config_path)

        print("Classifier config path:", self.classifier_config_path)
        print("Classifier checkpoint path:", self.classifier_ckpt_path)

        self.classifier_hash_folder = Path(self.classifier_ckpt_path).parent.name
        self.save_classifier_results_folder = os.path.join(
            self.experiment_folder_path,
            f"results_{self.inference_anomaly_detection_config.split}",
            f"image_classifier_results_checkpoint_{self.classifier_hash_folder}",
        )
        os.makedirs(self.save_classifier_results_folder, exist_ok=False)

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)

        self.evaluation_dataset_path = (
            f"{STORAGE_PARENT_FOLDER_PATH}/"
            f"{self.inference_anomaly_detection_config.dataset_name}/"
            f"{self.inference_anomaly_detection_config.split}"
        )

        print("ijepa_train_config_path:", self.ijepa_train_config_path)
        print("ijepa_ckpt_file_path:", self.ijepa_ckpt_file_path)

    def _load_evaluation_dataset(self):
        image_transform = transforms.Compose([
            transforms.Resize(
                (self.ijepa_train_config.crop_size, self.ijepa_train_config.crop_size), Image.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.ijepa_train_config.transform_mean,
                std=self.ijepa_train_config.transform_std,
            ),
        ])


        eval_dataset = LabelDataset(
            images_dir=self.evaluation_dataset_path,
            dataset_name=self.inference_anomaly_detection_config.dataset_name,
            count_border_pixels=self.ijepa_train_config.count_border_pixels,
            transform=image_transform,
        )

        eval_loader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        print("Number of evaluation samples:", len(eval_dataset))
        print("Number of evaluation batches:", len(eval_loader))

        assert eval_loader.batch_size == 1, "Error: please use only batch size = 1!"

        return eval_loader

    def run(self):
        with torch.inference_mode():
            # Init DinoV3/I-JEPA models and load target encoder from checkpoint
            dict_models = init_model_dinov3(
                device=self.device,
                encoder_model_name=self.ijepa_train_config.encoder_model_name,
                encoder_weights_path=self.ijepa_train_config.encoder_weights_source,
                predictor_depth=self.ijepa_train_config.predictor_depth,
                encoder_embed_dim=self.ijepa_train_config.encoder_embed_dim,
                predictor_embed_dim=self.ijepa_train_config.predictor_embed_dim,
                predictor_num_patches=(
                    self.ijepa_train_config.crop_size // self.ijepa_train_config.patch_size
                ) ** 2,
                predictor_num_heads=self.ijepa_train_config.predictor_num_heads,
                target_encoder_model_name=self.ijepa_train_config.target_encoder_model_name,
                target_encoder_weights_path=self.ijepa_train_config.target_encoder_weights_source,
            )

            print(f"Loading I-JEPA checkpoint from: {self.ijepa_ckpt_file_path}")
            loaded_dict = load_models_from_checkpoint(
                checkpoint_path=self.ijepa_ckpt_file_path,
                encoder=dict_models["encoder"],
                predictor=dict_models["predictor"],
                target_encoder=dict_models["target_encoder"],
                optimizer=None,
                scaler=None,
                dinov3_ckpt=True,
            )
            target_encoder = loaded_dict["target_encoder"]
            target_encoder.eval()

            # Load the trained classifier head
            classifier = HeadClassifier(
                embed_dim=self.ijepa_train_config.encoder_embed_dim,
                hidden_size=self.classifier_config.hidden_size,
                activation=self.classifier_config.activation,
            ).to(self.device)
            ckpt = torch.load(self.classifier_ckpt_path, map_location=self.device, weights_only=True)
            classifier.load_state_dict(ckpt["classifier_state_dict"])
            classifier.eval()

            eval_loader = self._load_evaluation_dataset()

            for _, (image, label, image_name, _) in enumerate(eval_loader):
                image = image.to(self.device)
                label = label.detach().cpu().item()
                image_name = image_name[0].split('.')[0]

                features = extract_features(target_encoder, image)
                logit = classifier(features)
                score = torch.sigmoid(logit).item()

                result = {"score": score, "label": label}
                save_path = os.path.join(
                    self.save_classifier_results_folder,
                    f"{image_name}_score_label_{label}.json",
                )
                with open(save_path, "w") as f:
                    json.dump(result, f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions using the synthetic anomaly classifier model"
    )
    parser.add_argument(
        "--inference_config_path", type=str, help="Inference configuration file (used for anomaly detection)"
    )
    args = parser.parse_args()

    generator = PredictionsClassifierModelGenerator(inference_config_path=args.inference_config_path)
    generator.run()


if __name__ == "__main__":
    main()
