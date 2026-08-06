import torch
import os
import glob
import json
import argparse
from torchvision import transforms
import torch.nn.functional as F

from src.utils.configs import *
from transformers import ViTConfig, ViTModel
from src.helper import (
    load_models_from_checkpoint,
    init_model_dinov3
)
from src.utils.anomaly_maps import *
from src.utils.preprocessing import *
from src.utils.dataset import *

from instance_specific_config import DATASET_PATH, ROOT_DIRECTORY_EXPERIMENTS_PATH

STORAGE_PARENT_FOLDER_PATH = DATASET_PATH


class PredictionsDistilledModelGenerator:
    def __init__(
            self, 
            inference_config_path: str,
            train_distilled_model_config_path: str
        ):
        # Read the inference configuration file for anomaly detection
        self.inference_anomaly_detection_config = read_inference_anomaly_detection_config(config_path=inference_config_path)
        self.distillation_model_path = train_distilled_model_config_path.replace("config.yaml", "student_model.pth")
        self.distillation_train_config_path = train_distilled_model_config_path


        self.experiment_folder_path = (
            f"{ROOT_DIRECTORY_EXPERIMENTS_PATH}/"
            f"experiments_{self.inference_anomaly_detection_config.dataset_name}/"
            f"{self.inference_anomaly_detection_config.experiment_name}"
        )


        self.distillation_experiment_folder_path = (
            f"{self.experiment_folder_path}/checkpoints_distilled_model"
        )

        self.distillation_train_config = read_train_distillation_config(config_path=self.distillation_train_config_path)

        ijepa_train_config_path = (
            f"{self.experiment_folder_path}/logs/config_run.yaml"
        )
        print("ijepa_train_config_path: ", ijepa_train_config_path)
        
        self.ijepa_train_config = read_train_ijepa_config(config_path=ijepa_train_config_path)
        ijepa_ckpt_folder_path = (
            f"{self.experiment_folder_path}/checkpoints"
        )
        self.ijepa_ckpt_file_path = os.path.join(
            ijepa_ckpt_folder_path,
            f"{self.inference_anomaly_detection_config.experiment_name}-epoch-20.pth.tar",
            # f"{self.inference_anomaly_detection_config.experiment_name}-latest.pth.tar",
        )
        print("I-Jepa checkpoint file path: ", self.ijepa_ckpt_file_path)
       
        self.distilled_model_hash_folder = self.distillation_train_config_path.split("/")[-2]
        self.save_per_patch_results_folder = (
            f"{self.experiment_folder_path}/results_{self.inference_anomaly_detection_config.split}/"
            f"image_per_patch_{self.inference_anomaly_detection_config.distance}_distances_"
            f"distilled_model_checkpoint_{self.distilled_model_hash_folder}/"
        )

        os.makedirs(self.save_per_patch_results_folder, exist_ok=True)

        self.evaluation_dataset_path = (
            f"{STORAGE_PARENT_FOLDER_PATH}/"
            f"{self.inference_anomaly_detection_config.dataset_name}/"
            f"{self.inference_anomaly_detection_config.split}"
        )

        # Set up the device
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(self.device)

        print("ijepa_train_config_path = ", ijepa_train_config_path)
        print("ijepa_ckpt_file_path", self.ijepa_ckpt_file_path)
        print("distillation_train_config_path: ", self.distillation_train_config_path)
        print("distillation_model_path: ", self.distillation_model_path)

    
    def _load_evaluation_dataset(self):
        image_transform = transforms.Compose([
            transforms.Resize((self.ijepa_train_config.crop_size, self.ijepa_train_config.crop_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.ijepa_train_config.transform_mean, 
                std=self.ijepa_train_config.transform_std),
        ])

        inverse_image_transform = transforms.Compose([
            transforms.Normalize(mean = [ 0.0, 0.0, 0.0],
                                std = [ 1.0 / self.ijepa_train_config.transform_std[0],
                                        1.0 / self.ijepa_train_config.transform_std[1], 
                                        1.0 / self.ijepa_train_config.transform_std[2]],
                                inplace=False),

            transforms.Normalize(mean = [-self.ijepa_train_config.transform_mean[0],
                                        -self.ijepa_train_config.transform_mean[1],
                                        -self.ijepa_train_config.transform_mean[2]],
                                std = [ 1.0, 1.0, 1.0],
                                inplace=False)
            ])
        
        eval_dataset = LabelDataset(
            images_dir=self.evaluation_dataset_path,
            dataset_name=self.inference_anomaly_detection_config.dataset_name,
            count_border_pixels=self.ijepa_train_config.count_border_pixels,
            transform=image_transform
        )

        eval_loader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False
        )

        print("Number of evaluation samples: ", len(eval_dataset))
        print("Number of evaluation batches: ", len(eval_loader))
        
        assert eval_loader.batch_size == 1, "Error: for generating random boxes, please use only batch size = 1! "
            
        return inverse_image_transform, eval_loader
        

    def run(self):
        with torch.inference_mode():
            # Init DinoV3/I-Jepa models: (context) encoder (DinoV3) + predictor (I-Jepa checkpoint) + target_encoder (DinoV3 - same architecture as the context encoder)
            dict_models = init_model_dinov3(
                device=self.device,
                encoder_model_name=self.ijepa_train_config.encoder_model_name,
                encoder_weights_path=self.ijepa_train_config.encoder_weights_source,
                predictor_depth=self.ijepa_train_config.predictor_depth,
                encoder_embed_dim=self.ijepa_train_config.encoder_embed_dim,
                predictor_embed_dim=self.ijepa_train_config.predictor_embed_dim,
                predictor_num_patches=(self.ijepa_train_config.crop_size // self.ijepa_train_config.patch_size) * (self.ijepa_train_config.crop_size // self.ijepa_train_config.patch_size),
                predictor_num_heads=self.ijepa_train_config.predictor_num_heads,
                target_encoder_model_name=self.ijepa_train_config.target_encoder_model_name,
                target_encoder_weights_path=self.ijepa_train_config.target_encoder_weights_source
            )
            encoder = dict_models["encoder"]
            predictor = dict_models["predictor"]
            target_encoder = dict_models["target_encoder"]

            # Load the pre-trained checkpoint (encoder + target encoder - DinoV3 VitS, predictor - IJepa checkpoint)
            print(f"Loading checkpoint from: {self.ijepa_ckpt_file_path}")
            loaded_dict_from_checkpoint = load_models_from_checkpoint(
                checkpoint_path=self.ijepa_ckpt_file_path,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                optimizer=None,
                scaler=None,
                dinov3_ckpt=True
            )

            target_encoder = loaded_dict_from_checkpoint["target_encoder"]

            # Loading the trained model
            configuration = ViTConfig(
                hidden_size=self.distillation_train_config.hidden_size,
                num_hidden_layers=self.distillation_train_config.num_hidden_layers,
                num_attention_heads=self.distillation_train_config.num_attention_heads,
                intermediate_size=self.distillation_train_config.intermediate_size
            )

            student_model = ViTModel(configuration).to(self.device)
            print("Distillation model path: ", self.distillation_model_path)
            student_model.load_state_dict(torch.load(self.distillation_model_path, weights_only=True))

            _, eval_loader = self._load_evaluation_dataset()

            target_encoder.eval()
            student_model.eval()
            
            image_per_patch_distances = {}
            for _, (image, label, image_name, _) in enumerate(eval_loader):
                image = image.to(self.device)
                label = label.detach().cpu().item()
                image_name = image_name[0].split('.')[0]
                if "camelyon16_256" in self.distillation_train_config.dataset_name:
                    # In Camelyon16_256 dataset, some images of different types (anomaly, normal) have the same name. Hence, the image_name won't be a unique identifier in that case.
                    # We will append the label to the image name in order to distinguish between the anomaly/normal images that have identical name.
                    image_name = f"{image_name}_{label}"

                with torch.no_grad():
                    # --- Target encoder (teacher) forward prop
                    target_encoder_emb = target_encoder(image)
                    if self.distillation_train_config.layer_norm:
                        target_encoder_emb = F.layer_norm(target_encoder_emb, (target_encoder_emb.size(-1),))  # normalize over feature-dim
                        
                    # --- Student model forward prop
                    student_emb = student_model(image).last_hidden_state[:, 1: , :]
                    if self.distillation_train_config.layer_norm:
                        student_emb = F.layer_norm(student_emb, (student_emb.size(-1),))  # normalize over feature-dim

                    assert target_encoder_emb.shape == student_emb.shape

                    if self.distillation_train_config.normalize_before_computing_distance:
                        target_encoder_emb = F.normalize(target_encoder_emb, dim=-1)
                        student_emb = F.normalize(student_emb, dim=-1)

                    # --- Compute the metrics
                    # Average distance between the predicted and target patch-level representations (embeddings) for the random box
                    if self.inference_anomaly_detection_config.distance == "l2":
                        distance_per_patch = torch.linalg.vector_norm(student_emb - target_encoder_emb, ord=2, dim=2)
                    elif self.inference_anomaly_detection_config.distance == "l1":
                        distance_per_patch = torch.linalg.vector_norm(student_emb - target_encoder_emb, ord=1, dim=2)
                    else:
                        print("Distance metric is not correct! Set DISTANCE to one of the following options: l1, l2")
                        break
                    
                    num_patches = distance_per_patch.shape[1]
                    image_per_patch_distances[image_name] = {patch_idx: [] for patch_idx in range(num_patches)}
                    for patch_idx in range(num_patches):
                        image_per_patch_distances[image_name][patch_idx] = [distance_per_patch[0, patch_idx].item()]

                    save_dict_path = os.path.join(
                        self.save_per_patch_results_folder,
                        image_name + "_distances_label" + "_" + str(label) + ".json"
                    )
                    with open(save_dict_path, "w") as f:
                        json.dump(image_per_patch_distances[image_name], f)
                    

def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Generate predictions with a pretrained I-Jepa system on randomly sampled boxes")
    parser.add_argument(
        "--inference_config_path", type=str, help="Inference configuration file (used for anomaly detection)",
    )
    parser.add_argument(
        "--train_distilled_model_config_path", type=str, help="Configuration used for training the distilled model"
    )

    args = parser.parse_args()

    predictions_random_boxes_generator = PredictionsDistilledModelGenerator(
        inference_config_path=args.inference_config_path,
        train_distilled_model_config_path=args.train_distilled_model_config_path
    )

    predictions_random_boxes_generator.run()


if __name__ == "__main__":
    main()

