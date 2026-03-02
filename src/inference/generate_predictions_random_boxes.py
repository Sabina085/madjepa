import torch
import yaml
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from torchvision import transforms

from src.utils.inference import *
from src.utils.dataset import *
from src.helper import (
    load_models_from_checkpoint,
    init_model_dinov3
)
from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch
from src.masks.multiblock import MaskCollator

subpath_list = os.getcwd().split(os.sep)
root_directory_path = os.path.join(*subpath_list[:-1])
current_dir = subpath_list[-1]

###  To modify these configuration values
DATASET_NAME = "BraTS2021_slice"
EXPERIMENT_NAME = "enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3_cropped_random_crop_training"
CONFIG_NAME = "BraTS2021_slice_enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3_imagenet_normalization.yaml"
SPLIT = "valid"
# How many random target boxes to sample per image
NUM_TARGET_BOXES_PER_IMAGE = 30
DISTANCE = "l2"
# If you want to save the masked images using the (non essential - useful for debugging and for understanding how the box sampling is performed)
SAVE_IMAGES_WITH_MASKS = True
###

EVALUATION_DATASET_PATH = f"/media/vhondru/hdd/istoian/BMAD/dataset/{DATASET_NAME}/{SPLIT}"

EXPERIMENT_FOLDER_PATH = (
    f"/{root_directory_path}/{current_dir}/"
    f"experiments_{DATASET_NAME}/{EXPERIMENT_NAME}/results_{SPLIT}"
)

CONFIG_PATH = f"/{root_directory_path}/{current_dir}/configs/{CONFIG_NAME}"

CKPT_FOLDER_PATH = EXPERIMENT_FOLDER_PATH.replace(f"results_{SPLIT}", "checkpoints")
CKPT_FILE_PATH = f"{CKPT_FOLDER_PATH}/{EXPERIMENT_NAME}-latest.pth.tar"

RANDOM_BOXES_PATH = f"{EXPERIMENT_FOLDER_PATH}/random_boxes_images"

DISTANCES_RESULTS_CSV_PATH = (
    f"{EXPERIMENT_FOLDER_PATH}/"
    f"{DISTANCE}_distance_random_box_{NUM_TARGET_BOXES_PER_IMAGE}_seeds.csv"
)

SAVE_PER_PATCH_RESULTS_FOLDER = (
    f"{EXPERIMENT_FOLDER_PATH}/"
    f"image_per_patch_{DISTANCE}_distances_{NUM_TARGET_BOXES_PER_IMAGE}_seeds/"
)

SAVE_CROPPED_BORDERS_DICT_FOLDER = (
    f"{EXPERIMENT_FOLDER_PATH}/"
    "cropped_borders_dicts/"
)


def main():
    # Set up the device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Read the configuration file (which was used for training the checkpoint)
    # --- context encoder
    encoder_model_name = config["meta"]["encoder_model_name"]
    encoder_weights_path = config["meta"]["encoder_weights_source"]
    encoder_embed_dim = config["meta"]["encoder_embed_dim"]

    # --- target encoder
    target_encoder_model_name = config["meta"]["target_encoder_model_name"]
    target_encoder_weights_path = config["meta"]["target_encoder_weights_source"]

    # --- predictor
    predictor_depth = config["meta"]["predictor_depth"]
    predictor_num_heads = config["meta"]["predictor_num_heads"]
    predictor_embed_dim = config["meta"]["predictor_embed_dim"]

    # --- data transforms
    crop_size = config["data"]["crop_size"]
    patch_size = config["mask"]["patch_size"]

    # -- target blocks / masks
    target_block_scale = config["mask"]["pred_mask_scale"]
    target_block_aspect_ratio = config["mask"]["pred_mask_aspect_ratio"]

    # --- context block / mask
    enc_mask_scale = config["mask"]["enc_mask_scale"]
    enc_mask_aspect_ratio = config["mask"]["enc_mask_aspect_ratio"]
    enc_min_keep = config["mask"]["enc_min_keep"]

    # Init DinoV3/I-Jepa models: (context) encoder (DinoV3) + predictor (I-Jepa checkpoint) + target_encoder (DinoV3 - same architecture as the context encoder)
    dict_models = init_model_dinov3(
        device=device,
        encoder_model_name=encoder_model_name,
        encoder_weights_path=encoder_weights_path,
        predictor_depth=predictor_depth,
        encoder_embed_dim=encoder_embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        predictor_num_patches=(crop_size // patch_size) * (crop_size // patch_size),
        predictor_num_heads=predictor_num_heads,
        target_encoder_model_name=target_encoder_model_name,
        target_encoder_weights_path=target_encoder_weights_path
    )
    encoder = dict_models["encoder"]
    predictor = dict_models["predictor"]
    target_encoder = dict_models["target_encoder"]

    # Load the pre-trained checkpoint (encoder + target encoder - DinoV3 VitS, predictor - IJepa checkpoint)
    print(f"Loading checkpoint from: {CKPT_FILE_PATH}")
    loaded_dict_from_checkpoint = load_models_from_checkpoint(
        checkpoint_path=CKPT_FILE_PATH,
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        optimizer=None,
        scaler=None,
        dinov3_ckpt=True
    )
    encoder = loaded_dict_from_checkpoint["encoder"]
    predictor = loaded_dict_from_checkpoint["predictor"]
    target_encoder = loaded_dict_from_checkpoint["target_encoder"]

    del loaded_dict_from_checkpoint

    # Load the validation dataset
    image_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size), Image.BICUBIC),
        transforms.ToTensor(),
        # BrainMRI statistics
        # transforms.Normalize(mean=(0.1129, 0.1240, 0.1445), std=(0.0392, 0.0495, 0.0612)),
        # ImageNet statistics
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    mask_transform = transforms.Compose([
        # transforms.Resize((crop_size, crop_size), Image.NEAREST),
        transforms.ToTensor()
    ])

    inverse_image_transform = transforms.Compose([
        transforms.Normalize(mean = [ 0.0, 0.0, 0.0],
                             std = [ 1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225],
                             inplace=False),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406],
                             std = [ 1.0, 1.0, 1.0],
                             inplace=False)
        ])
    
    eval_dataset = SegmentationWithLabelDataset(
        images_dir=EVALUATION_DATASET_PATH,
        transform=image_transform,
        mask_transform=mask_transform)

    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=1,
        shuffle=False)
    
    print("Number of evaluation samples: ", len(eval_loader))

    # Used for sampling the size of a target block
    mc = MaskCollator(
        input_size=(crop_size, crop_size),
        patch_size=patch_size
    )
    g = torch.Generator()

    list_box_sizes = []
    list_image_names = []
    list_segmentation_masks = []
    list_mean_distances_per_image = []
    list_max_distances_per_image = []
    list_labels = []
    list_seeds = []
    list_cropped_borders_dicts = []

    # -- used for building the anomaly map
    num_patches = (crop_size // patch_size) * (crop_size // patch_size)
    image_per_patch_distances = {}
    # --
    
    for box_idx in tqdm(range(NUM_TARGET_BOXES_PER_IMAGE)):
        seed = box_idx
        g.manual_seed(seed)

        target_block_h, target_block_w = mc._sample_block_size(
            generator=g,
            scale=target_block_scale,
            aspect_ratio_scale=target_block_aspect_ratio)
        
        for _, (image, segmentation_mask, label, image_name, borders_dict) in enumerate(eval_loader):
            image = image.to(device)
            label = label.to(device)
            image_name = image_name[0].split('.')[0]
            segmentation_mask = np.squeeze(np.asarray(segmentation_mask))

            if box_idx == 0:
                image_per_patch_distances[image_name] = {patch_idx: [] for patch_idx in range(num_patches)}

            dict_mask_target_encoder = get_random_box(
                image_height=crop_size,
                image_width=crop_size,
                box_height=target_block_h,
                box_width=target_block_w,
                patch_size=patch_size,
                seed=seed
            )
            random_box = dict_mask_target_encoder["random_box"]

            random_mask_target_encoder = transform_from_target_box_to_mask(
                image_height=crop_size,
                image_width=crop_size,
                box=random_box,
                patch_size=patch_size
            )["mask"]

            list_masks_target_encoder = [torch.unsqueeze(random_mask_target_encoder, 0).to(device=device)]
            
            encoder_mask = [sample_context(
                enc_mask_scale=enc_mask_scale,
                aspect_ratio=enc_mask_aspect_ratio,
                image_height=crop_size,
                image_width=crop_size,
                patch_size=patch_size,
                target_masks=[random_mask_target_encoder],
                min_keep=enc_min_keep
            ).to(device)]

            if SAVE_IMAGES_WITH_MASKS and (seed == 0): # save examples of sampled boxes / images only for the first seed
                # Save the image using the encoder context mask
                encoder_context_mask_arr = transform_from_mask_to_box(
                    image_height=crop_size,
                    image_width=crop_size,
                    mask=encoder_mask[0],
                    patch_size=patch_size
                )
                
                if not os.path.exists(RANDOM_BOXES_PATH):
                    os.makedirs(RANDOM_BOXES_PATH, exist_ok=True)

                ### Code used for saving the generated input boxes (not essential, can be removed)
                # Saving the original input image (unmasked)
                image_arr = inverse_image_transform(image[0, :, :, :]).detach().cpu().numpy()
                Image.fromarray((255 * np.transpose(image_arr, (1, 2, 0)))\
                    .astype("uint8")).convert("RGB")\
                    .save(f"{RANDOM_BOXES_PATH}/{image_name}_image.png")
                Image.fromarray((255 * np.transpose(image[0, :, :, :].detach().cpu().numpy(), (1, 2, 0)))\
                    .astype("uint8")).convert("RGB")\
                    .save(f"{RANDOM_BOXES_PATH}/{image_name}_normalized_image.png")
    
                # Saving the input image masked using the sampled random box (only the pixels inside the random box are visible)
                pixels_random_mask = np.repeat(np.expand_dims(dict_mask_target_encoder["pixels_random_mask"], axis=0), 3, axis=0)\
                    .astype("uint8")
                masked_image = np.transpose((255 * image_arr * pixels_random_mask)\
                    .astype(np.uint8), (1, 2, 0))
                Image.fromarray(masked_image).convert("RGB")\
                    .save(f"{RANDOM_BOXES_PATH}/{image_name}_random_pixels_mask_image_seed_{seed}.png")

                # Save the image with the encoder masked context
                encoder_context_mask = np.repeat(np.expand_dims(encoder_context_mask_arr, axis=0), 3, axis=0).astype("uint8")
                masked_image = np.transpose((255 * image_arr * encoder_context_mask)\
                    .astype(np.uint8), (1, 2, 0))
                Image.fromarray(masked_image).convert("RGB")\
                    .save(f"{RANDOM_BOXES_PATH}/{image_name}_context_image_seed_{seed}.png")
                ###
        
            encoder.eval()
            target_encoder.eval()
            predictor.eval()

            with torch.no_grad():
                # --- Target encoder forward prop
                # Forward propagation on the original input image (unmasked, as it was mentioned in the paper)
                target_encoder_emb = target_encoder.forward(image)
                # We do not need this anymore as we're already applying a layer normalization at the end of the target encoder forward function
                # target_encoder_emb = F.layer_norm(target_encoder_emb, (target_encoder_emb.size(-1),))  # normalize over feature-dim
                # retrieve the batch size
                B = len(target_encoder_emb)
                
                # Retrieve the embeddings only for the mask patches 
                target_encoder_emb = apply_masks(target_encoder_emb, list_masks_target_encoder)
                target_encoder_emb = repeat_interleave_batch(target_encoder_emb, B, repeat=1)

                # --- (Context) Encoder forward prop
                encoder_emb = encoder.forward(image, encoder_mask)

                # --- Predictor forward prop
                predictor_emb = predictor.forward(encoder_emb, encoder_mask, list_masks_target_encoder)

                # --- Compute the metrics
                # Average distance between the predicted and target patch-level representations (embeddings) for the random box
                if DISTANCE == "l2":
                    distance_per_embedding = torch.linalg.vector_norm(predictor_emb - target_encoder_emb, ord=2, dim=2)
                elif DISTANCE == "l1":
                    distance_per_embedding = torch.linalg.vector_norm(predictor_emb - target_encoder_emb, ord=1, dim=2)
                else:
                    print("Distance metric is not correct! Set DISTANCE to one of the following options: l1, l2")
                    return          

                mean_distance_per_image = torch.mean(input=distance_per_embedding, dim=1, keepdim=False)
                list_mean_distances_per_image.append(mean_distance_per_image.detach().item())

                max_distance_per_image = torch.max(input=distance_per_embedding, dim=1, keepdim=False)
                list_max_distances_per_image.append(max_distance_per_image.values.detach().item())

                list_image_names.append(image_name)
                list_box_sizes.append((target_block_h * patch_size, target_block_w * patch_size))
                list_labels.append(label.detach().item())
                list_seeds.append(seed)
                list_segmentation_masks.append(segmentation_mask)
                list_cropped_borders_dicts.append(borders_dict)

                # Get distances per patch embedding for building an anomaly map
                if DISTANCE == "l2":
                    distance_per_patch = torch.linalg.vector_norm(predictor_emb - target_encoder_emb, ord=2, dim=2)
                elif DISTANCE == "l1":
                    distance_per_patch = torch.linalg.vector_norm(predictor_emb - target_encoder_emb, ord=1, dim=2)
                else:
                    print("Distance metric is not correct! Set DISTANCE to one of the following options: l1, l2")
                    return
                
                for mask in list_masks_target_encoder:
                    mask = torch.squeeze(mask)
                    indices_mask_target = mask.tolist()
                    for idx_tensor, index_patch in enumerate(indices_mask_target):
                        image_per_patch_distances[image_name][index_patch] += [distance_per_patch[0, idx_tensor].item()]

        try:
            print("seed: ", seed)
            print(f"Mean {DISTANCE} anomaly images mean: ", np.mean(
                [distance for distance, label in zip(list_mean_distances_per_image, list_labels) if label == 1]))
            print(f"Mean {DISTANCE} normal images mean: ", np.mean(
                [distance for distance, label in zip(list_mean_distances_per_image, list_labels) if label == 0]))
            print(f"Mean {DISTANCE} anomaly images max: ", np.mean(
                [distance for distance, label in zip(list_max_distances_per_image, list_labels) if label == 1]))
            print(f"Mean {DISTANCE} normal images max: ", np.mean(
                [distance for distance, label in zip(list_max_distances_per_image, list_labels) if label == 0]))
        except Exception as e:
            print(f"Error when computing the mean distance per image: {e}")

    df = pd.DataFrame({
        "image_name": list_image_names,
        "label": list_labels,
        f"mean_{DISTANCE}_distance": list_mean_distances_per_image,
        f"max_{DISTANCE}_distance": list_max_distances_per_image,
        "box_size": list_box_sizes,
        "seed": list_seeds
    })
    
    if not os.path.exists(Path(DISTANCES_RESULTS_CSV_PATH).parent):
        os.makedirs(Path(DISTANCES_RESULTS_CSV_PATH).parent, exist_ok=True)

    df.to_csv(DISTANCES_RESULTS_CSV_PATH, index=False)

    if not os.path.exists(SAVE_PER_PATCH_RESULTS_FOLDER):
        os.makedirs(SAVE_PER_PATCH_RESULTS_FOLDER, exist_ok=True)
    
    if not os.path.exists(SAVE_CROPPED_BORDERS_DICT_FOLDER):
        os.makedirs(SAVE_CROPPED_BORDERS_DICT_FOLDER, exist_ok=True)

    for image_name, label, segmentation_mask, cropped_border_dict in zip(list_image_names, list_labels, list_segmentation_masks, list_cropped_borders_dicts):

        save_dict_path = os.path.join(SAVE_PER_PATCH_RESULTS_FOLDER, image_name + "_distances_label" + "_" + str(label) + ".json")
        with open(save_dict_path, "w") as f:
            json.dump(image_per_patch_distances[image_name], f)
        
        save_borders_dict_path = os.path.join(SAVE_CROPPED_BORDERS_DICT_FOLDER, image_name + "_cropped_borders_dict_label" + "_" + str(label) + ".json")
        with open(save_borders_dict_path, "w") as f:
            cropped_border_dict = {k: v.item() for k, v in cropped_border_dict.items()}
            json.dump(cropped_border_dict, f)

        save_segmentation_path = os.path.join(SAVE_PER_PATCH_RESULTS_FOLDER, image_name + "_segmentation_mask_label_" + str(label) + ".npy")
        np.save(file=save_segmentation_path, arr=segmentation_mask)


if __name__ == "__main__":
    main()
