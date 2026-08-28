import argparse
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from torchvision import transforms
import torch.nn.functional as F

from src.utils.inference import *
from src.utils.dataset import *
from src.utils.configs import *
from src.helper import (
    load_models_from_checkpoint,
    init_model_dinov3
)
from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch
from src.masks.multiblock import MaskCollator

from instance_specific_config import DATASET_PATH, ROOT_DIRECTORY_EXPERIMENTS_PATH

STORAGE_PARENT_FOLDER_PATH = DATASET_PATH

print(STORAGE_PARENT_FOLDER_PATH)


class PredictionsRandomBoxesGenerator:
    def __init__(self, inference_config_path: str):
        # Read the inference configuration file for anomaly detection
        self.inference_config = read_inference_anomaly_detection_config(config_path=inference_config_path)

        self.evaluation_dataset_path = (
            f"{STORAGE_PARENT_FOLDER_PATH}/"
            f"{self.inference_config.dataset_name}/{self.inference_config.split}"
        )

        self.experiment_folder_path = (
            f"{ROOT_DIRECTORY_EXPERIMENTS_PATH}/"
            f"experiments_{self.inference_config.dataset_name}/"
            f"{self.inference_config.experiment_name}/"
            f"results_{self.inference_config.split}"
        )

        self.train_config_path = (
            f"{ROOT_DIRECTORY_EXPERIMENTS_PATH}/"
            f"experiments_{self.inference_config.dataset_name}/"
            f"{self.inference_config.experiment_name}/logs/config_run.yaml"
        )


        # Read the ijepa training configuration file (which was used for training the checkpoint)
        self.train_config = read_train_ijepa_config(config_path=self.train_config_path)

        self.ckpt_folder_path = self.experiment_folder_path.replace(
            f"results_{self.inference_config.split}", "checkpoints"
        )
        self.ckpt_file_path = (
            f"{self.ckpt_folder_path}/"
            f"{self.inference_config.experiment_name}-epoch-20.pth.tar"
            # f"{self.inference_config.experiment_name}-latest.pth.tar"
        )

        self.random_boxes_path = f"{self.experiment_folder_path}/random_boxes_images"

        self.distances_results_csv_path = (
            f"{self.experiment_folder_path}/"
            f"{self.inference_config.distance}_distance_random_box_"
            f"{self.inference_config.num_sampled_random_boxes_per_image}_seeds.csv"
        )

        self.save_per_patch_results_folder = (
            f"{self.experiment_folder_path}/"
            f"image_per_patch_{self.inference_config.distance}_distances_"
            f"{self.inference_config.num_sampled_random_boxes_per_image}_seeds_"
            f"global_seed_{self.inference_config.global_seed_random_boxes}/"
        )

        self.save_cropped_borders_dict_folder = (
            f"{self.experiment_folder_path}/"
            "cropped_borders_dicts/"
        )

        # Set up the device
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)

    def load_evaluation_dataset(self):
            
        image_transform = transforms.Compose([
            transforms.Resize((self.train_config.crop_size, self.train_config.crop_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.train_config.transform_mean, std=self.train_config.transform_std),
        ])

        inverse_image_transform = transforms.Compose([
            transforms.Normalize(mean = [ 0.0, 0.0, 0.0],
                                std = [ 1.0 / self.train_config.transform_std[0],
                                        1.0 / self.train_config.transform_std[1], 
                                        1.0 / self.train_config.transform_std[2]],
                                inplace=False),

            transforms.Normalize(mean = [-self.train_config.transform_mean[0],
                                        -self.train_config.transform_mean[1],
                                        -self.train_config.transform_mean[2]],
                                std = [ 1.0, 1.0, 1.0],
                                inplace=False)
            ])
        
        eval_dataset = LabelDataset(
            images_dir=self.evaluation_dataset_path,
            dataset_name=self.inference_config.dataset_name,
            count_border_pixels=self.train_config.count_border_pixels,
            transform=image_transform)

        eval_loader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=1,
            shuffle=False)
        
        assert eval_loader.batch_size == 1, "Error: for generating random boxes, please use only batch size = 1! "
            
        return inverse_image_transform, eval_loader

    def save_results(
            self,
            list_image_names,
            list_labels,
            list_cropped_borders_dicts,
            image_per_patch_distances,
            list_mean_distances_per_image, 
            list_max_distances_per_image, 
            list_box_sizes, 
            list_seeds, 
    ):
        
        df = pd.DataFrame({
            "image_name": list_image_names,
            "label": list_labels,
            f"mean_{self.inference_config.distance}_distance": list_mean_distances_per_image,
            f"max_{self.inference_config.distance}_distance": list_max_distances_per_image,
            "box_size": list_box_sizes,
            "seed": list_seeds
        })
        if not os.path.exists(Path(self.distances_results_csv_path).parent):
            os.makedirs(Path(self.distances_results_csv_path).parent, exist_ok=True)

        df.to_csv(self.distances_results_csv_path, index=False)

        if not os.path.exists(self.save_per_patch_results_folder):
            os.makedirs(self.save_per_patch_results_folder, exist_ok=True)
        
        if not os.path.exists(self.save_cropped_borders_dict_folder):
            os.makedirs(self.save_cropped_borders_dict_folder, exist_ok=True)

        for image_name, label, cropped_border_dict in zip(list_image_names, list_labels, list_cropped_borders_dicts):
            save_dict_path = os.path.join(
                self.save_per_patch_results_folder,
                image_name + "_distances_label" + "_" + str(label) + ".json"
            )
            with open(save_dict_path, "w") as f:
                json.dump(image_per_patch_distances[image_name], f)
            
            save_borders_dict_path = os.path.join(
                self.save_cropped_borders_dict_folder,
                image_name + "_cropped_borders_dict_label" + "_" + str(label) + ".json"
            )
            
            with open(save_borders_dict_path, "w") as f:
                cropped_border_dict = {k: v.item() for k, v in cropped_border_dict.items()}
                json.dump(cropped_border_dict, f)
    
    def save_example_of_image_with_mask(
            self, 
            inverse_image_transform, 
            image, image_name, 
            dict_mask_target_encoder, 
            encoder_mask
        ):
        # Optional function used for saving the sampled input boxes and the original images (not essential, can be removed)
        encoder_context_mask_arr = transform_from_mask_to_box(
                        image_height=self.train_config.crop_size,
                        image_width=self.train_config.crop_size,
                        mask=encoder_mask[0],
                        patch_size=self.train_config.patch_size
                    )
                    
        if not os.path.exists(self.random_boxes_path):
            os.makedirs(self.random_boxes_path, exist_ok=True)
                    
        # Saving the original input image (unmasked)
        # Image before applying normalization
        image_arr = inverse_image_transform(image[0, :, :, :]).detach().cpu().numpy()
        Image.fromarray((255 * np.transpose(image_arr, (1, 2, 0)))\
                        .astype("uint8")).convert("RGB")\
                        .save(f"{self.random_boxes_path}/{image_name}_image.png")
        
        # Image after applying normalization
        Image.fromarray((255 * np.transpose(image[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)))\
                        .astype("uint8")).convert("RGB")\
                        .save(f"{self.random_boxes_path}/{image_name}_normalized_image.png")
        
        # Saving the input image masked using the sampled random box (only the pixels inside the random box are visible)
        pixels_random_mask = np.repeat(np.expand_dims(dict_mask_target_encoder["pixels_random_mask"], axis=0), 3, axis=0)\
                        .astype("uint8")
        masked_image = np.transpose((255 * image_arr * pixels_random_mask)\
                        .astype(np.uint8), (1, 2, 0))
        Image.fromarray(masked_image).convert("RGB")\
                        .save(f"{self.random_boxes_path}/{image_name}_random_pixels_mask_image.png")

        # Save the image with the encoder masked context
        encoder_context_mask = np.repeat(np.expand_dims(encoder_context_mask_arr, axis=0), 3, axis=0).astype("uint8")
        masked_image = np.transpose((255 * image_arr * encoder_context_mask)\
                        .astype(np.uint8), (1, 2, 0))
        Image.fromarray(masked_image).convert("RGB")\
                        .save(f"{self.random_boxes_path}/{image_name}_context_image.png")

    def get_target_encoder_and_encoder_masks(
            self,
            image, 
            target_block_h, 
            target_block_w, 
            inverse_image_transform,
            image_name,
            seed
        ):
        
        dict_mask_target_encoder = get_random_box(
                    image_height=self.train_config.crop_size,
                    image_width=self.train_config.crop_size,
                    box_height=target_block_h,
                    box_width=target_block_w,
                    patch_size=self.train_config.patch_size,
                    seed=seed
        )
        random_box = dict_mask_target_encoder["random_box"]

        random_mask_target_encoder = transform_from_target_box_to_mask(
                    image_height=self.train_config.crop_size,
                    image_width=self.train_config.crop_size,
                    box=random_box,
                    patch_size=self.train_config.patch_size
                )["mask"]

        list_masks_target_encoder = [torch.unsqueeze(random_mask_target_encoder, 0).to(device=self.device)]
                
        encoder_mask = [sample_context(
                    enc_mask_scale=self.train_config.enc_mask_scale,
                    aspect_ratio=self.train_config.enc_mask_aspect_ratio,
                    image_height=self.train_config.crop_size,
                    image_width=self.train_config.crop_size,
                    patch_size=self.train_config.patch_size,
                    target_masks=[random_mask_target_encoder],
                    min_keep=self.train_config.enc_min_keep
                ).to(self.device)]

        if self.inference_config.save_images_with_masks and (seed == 0): # save examples of sampled boxes / images only for the first seed
            self.save_example_of_image_with_mask(
                image=image,
                dict_mask_target_encoder=dict_mask_target_encoder,
                encoder_mask=encoder_mask,
                image_name=image_name,
                inverse_image_transform=inverse_image_transform
            )

        return list_masks_target_encoder, encoder_mask

    def log_distances_stats(
            self,
            list_mean_distances_per_image, 
            list_max_distances_per_image, 
            list_labels, 
            seed
        ):

        try:
            print("seed: ", seed)
            print(f"Current mean {self.inference_config.distance} anomaly images mean: ", np.mean(
                    [distance for distance, label in zip(list_mean_distances_per_image, list_labels) if label == 1]))
            print(f"Current mean {self.inference_config.distance} normal images mean: ", np.mean(
                    [distance for distance, label in zip(list_mean_distances_per_image, list_labels) if label == 0]))
            print(f"Current mean {self.inference_config.distance} anomaly images max: ", np.mean(
                    [distance for distance, label in zip(list_max_distances_per_image, list_labels) if label == 1]))
            print(f"Current mean {self.inference_config.distance} normal images max: ", np.mean(
                    [distance for distance, label in zip(list_max_distances_per_image, list_labels) if label == 0]))
        except Exception as e:
            print(f"Error when computing the mean distance per image: {e}")

    def run(self):
        with torch.inference_mode():
            # Init DinoV3/I-Jepa models: (context) encoder (DinoV3) + predictor (I-Jepa checkpoint) + target_encoder (DinoV3 - same architecture as the context encoder)
            dict_models = init_model_dinov3(
                device=self.device,
                encoder_model_name=self.train_config.encoder_model_name,
                encoder_weights_path=self.train_config.encoder_weights_source,
                predictor_depth=self.train_config.predictor_depth,
                encoder_embed_dim=self.train_config.encoder_embed_dim,
                predictor_embed_dim=self.train_config.predictor_embed_dim,
                predictor_num_patches=(self.train_config.crop_size // self.train_config.patch_size) * (self.train_config.crop_size // self.train_config.patch_size),
                predictor_num_heads=self.train_config.predictor_num_heads,
                target_encoder_model_name=self.train_config.target_encoder_model_name,
                target_encoder_weights_path=self.train_config.target_encoder_weights_source
            )

            # Load the pre-trained checkpoint (encoder + target encoder - DinoV3 VitS, predictor - IJepa checkpoint)
            print(f"Loading checkpoint from: {self.ckpt_file_path}")
            checkpoint_models = load_models_from_checkpoint(
                checkpoint_path=self.ckpt_file_path,
                encoder=dict_models["encoder"],
                predictor=dict_models["predictor"],
                target_encoder=dict_models["target_encoder"],
                optimizer=None,
                scaler=None,
                dinov3_ckpt=True
            )
            encoder = checkpoint_models["encoder"]
            predictor = checkpoint_models["predictor"]
            target_encoder = checkpoint_models["target_encoder"]

            inverse_image_transform, eval_loader = self.load_evaluation_dataset()
            
            print("Number of evaluation samples: ", len(eval_loader))

            # Used for sampling the size of a target block
            mask_collator = MaskCollator(
                input_size=(self.train_config.crop_size, self.train_config.crop_size),
                patch_size=self.train_config.patch_size
            )
            torch_generator = torch.Generator()

            list_box_sizes = []
            list_image_names = []
            list_mean_distances_per_image = []
            list_max_distances_per_image = []
            list_labels = []
            list_seeds = []
            list_cropped_borders_dicts = []

            # used for building the anomaly maps
            num_patches = (self.train_config.crop_size // self.train_config.patch_size) * (self.train_config.crop_size // self.train_config.patch_size)
            image_per_patch_distances = {}
            
            encoder.eval()
            target_encoder.eval()
            predictor.eval()

            for box_idx in tqdm(range(self.inference_config.num_sampled_random_boxes_per_image)):
                seed = box_idx + self.inference_config.global_seed_random_boxes
                torch_generator.manual_seed(seed)

                target_block_h, target_block_w = mask_collator._sample_block_size(
                    generator=torch_generator,
                    scale=self.train_config.target_block_scale,
                    aspect_ratio_scale=self.train_config.target_block_aspect_ratio)
                
                for _, (image, label, image_name, borders_dict) in enumerate(eval_loader):
                    image = image.to(self.device)
                    label = label.detach().item()
                    image_name = image_name[0].split('.')[0]
                    if "camelyon16_256" in self.train_config.image_folder:
                        # In Camelyon16_256 dataset, some images of different types (anomaly, normal) have the same name. Hence, the image_name won't be a unique identifier in that case.
                        # We will append the label to the image name in order to distinguish between the anomaly/normal images that have identical name.
                        image_name = f"{image_name}_{label}"

                    if box_idx == 0:
                        image_per_patch_distances[image_name] = {patch_idx: [] for patch_idx in range(num_patches)}

                    list_masks_target_encoder, encoder_mask = self.get_target_encoder_and_encoder_masks(
                        image=image,
                        target_block_h=target_block_h,
                        target_block_w=target_block_w,
                        inverse_image_transform=inverse_image_transform,
                        image_name=image_name,
                        seed=seed
                    )
                    with torch.no_grad():      
                        # Target encoder forward prop
                        # Forward propagation on the original input image (unmasked, as it was mentioned in the paper)
                        target_encoder_emb = target_encoder(image)
                        if self.train_config.layer_norm_teacher:
                            target_encoder_emb = F.layer_norm(target_encoder_emb, (target_encoder_emb.size(-1),))  # normalize over feature-dim
                        # retrieve the batch size
                        B = len(target_encoder_emb)
                        
                        # Retrieve the embeddings only for the mask patches 
                        target_encoder_emb = apply_masks(target_encoder_emb, list_masks_target_encoder)
                        target_encoder_emb = repeat_interleave_batch(target_encoder_emb, B, repeat=1)

                        # Context encoder forward prop
                        encoder_emb = encoder(image, encoder_mask)

                        # Predictor forward prop
                        predictor_emb = predictor(encoder_emb, encoder_mask, list_masks_target_encoder)

                        # Compute the metrics
                        # Average distance between the predicted and target patch-level representations (embeddings) for the random box
                        # Get distances per patch embedding for building an anomaly map
                        if self.inference_config.distance == "l2":
                            distance_per_patch = torch.linalg.vector_norm(predictor_emb - target_encoder_emb, ord=2, dim=2)
                        elif self.inference_config.distance == "l1":
                            distance_per_patch = torch.linalg.vector_norm(predictor_emb - target_encoder_emb, ord=1, dim=2)
                        else:
                            print("Distance metric is not correct! Set DISTANCE to one of the following options: l1, l2")
                            return        

                        mean_distance_per_image = torch.mean(input=distance_per_patch, dim=1, keepdim=False)
                        list_mean_distances_per_image.append(mean_distance_per_image.detach().item())

                        max_distance_per_image = torch.max(input=distance_per_patch, dim=1, keepdim=False)
                        list_max_distances_per_image.append(max_distance_per_image.values.detach().item())

                        list_image_names.append(image_name)
                        list_box_sizes.append((target_block_h * self.train_config.patch_size, target_block_w * self.train_config.patch_size))
                        list_labels.append(label)
                        list_seeds.append(seed)
                        list_cropped_borders_dicts.append(borders_dict)

                        mask = torch.squeeze(list_masks_target_encoder[0])
                        indices_mask_target = mask.tolist()
                        for idx_tensor, index_patch in enumerate(indices_mask_target):
                            image_per_patch_distances[image_name][index_patch] += [distance_per_patch[0, idx_tensor].item()]

                self.log_distances_stats(
                    list_mean_distances_per_image=list_mean_distances_per_image,
                    list_max_distances_per_image=list_max_distances_per_image,
                    list_labels=list_labels,
                    seed=seed
                )
            
            self.save_results(
                image_per_patch_distances=image_per_patch_distances,
                list_image_names=list_image_names,
                list_labels=list_labels,
                list_cropped_borders_dicts=list_cropped_borders_dicts, 
                list_mean_distances_per_image=list_mean_distances_per_image,
                list_max_distances_per_image=list_max_distances_per_image,
                list_box_sizes=list_box_sizes,
                list_seeds=list_seeds
            )
        

def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Generate predictions with a pretrained IJepa system on randomly sampled boxes")
    parser.add_argument("--inference_config_path", type=str, help="Inference configuration file (used for anomaly detection)")

    args = parser.parse_args()

    predictions_random_boxes_generator = PredictionsRandomBoxesGenerator(inference_config_path=args.inference_config_path)
    predictions_random_boxes_generator.run()


if __name__ == "__main__":
    main()
