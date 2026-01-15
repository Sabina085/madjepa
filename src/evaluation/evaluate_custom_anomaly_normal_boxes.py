import torch
import statistics
import os
import yaml
import numpy as np
import pandas as pd
from PIL import Image

import torch.nn.functional as F
from torchvision import transforms

from src.utils.utils_evaluation import *
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

CONFIG_PATH = f"/{root_directory_path}/{current_dir}/configs/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3.yaml"
CKPT_PATH = f"/{root_directory_path}/{current_dir}/experiments/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3/checkpoints/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3-latest.pth.tar"
VALIDATION_DATASET_PATH = f"/{root_directory_path}/BMAD/dataset/BraTS2021_slice/valid"
SELECTED_BOXES_PATH = f"/{root_directory_path}/{current_dir}/experiments/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3/custom_anomaly_normal_boxes"
RESULTS_CSV_PATH = f"/{root_directory_path}/{current_dir}/experiments/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3/l2_distance_anomaly_vs_normal_box_validation_multiple_seeds.csv"

# How many times to repeat the selection of the boxes (as it involves some randomness) 
NUM_SEEDS = 15
# If you want to save the masked images using the (non essential - useful for debugging and for understanding how the box sampling is performed)
SAVE_IMAGES_WITH_MASKS = True

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
    print(f"Loading checkpoint from: {CKPT_PATH}")
    loaded_dict_from_checkpoint = load_models_from_checkpoint(
        checkpoint_path=CKPT_PATH,
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

    # Load the validation dataset
    image_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1129, 0.1240, 0.1445), std=(0.0392, 0.0495, 0.0612)),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size), Image.NEAREST),
        transforms.ToTensor()
    ])
    inverse_image_transform = transforms.Compose([
        transforms.Normalize(mean = [ 0.0, 0.0, 0.0],
                             std = [ 1.0 / 0.0392, 1.0 / 0.0495, 1.0 / 0.0612],
                             inplace=False),
        transforms.Normalize(mean = [ -0.1129, -0.1240, -0.1445],
                             std = [ 1.0, 1.0, 1.0],
                             inplace=False)
        ])
    
    val_dataset = SegmentationWithLabelDataset(
        images_dir=VALIDATION_DATASET_PATH,
        transform=image_transform,
        mask_transform=mask_transform)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False)
    
    print("Total number of validation samples: ", len(val_loader))

    # Used for sampling the size of a target block
    mc = MaskCollator(
        input_size=(crop_size, crop_size),
        patch_size=patch_size
    )
    g = torch.Generator()

    list_l2_anomaly = []
    list_l2_normal = []
    list_images_names = []
    box_sizes = []

    for seed in range(NUM_SEEDS):
        g.manual_seed(seed)

        target_block_h, target_block_w = mc._sample_block_size(
            generator=g,
            scale=target_block_scale,
            aspect_ratio_scale=target_block_aspect_ratio
        )
        
        for _, (images, segmentation_masks, labels, image_name) in enumerate(val_loader):
            if labels[0].item() == 0:
                # Sample the two boxes (anomaly + normal) only from images with anomalies
                continue

            images = images.to(device)
            labels = labels.to(device)
            image_name = image_name[0].split('.')[0]
            segmentation_masks = np.squeeze(np.array(segmentation_masks))
            
            dict_masks_target_encoder = get_custom_target_anomaly_and_normal_box(
                segmentation_mask=segmentation_masks,
                box_height=target_block_h,
                box_width=target_block_w,
                patch_size=patch_size,
                seed=seed
            )

            if SAVE_IMAGES_WITH_MASKS:
                if not os.path.exists(SELECTED_BOXES_PATH):
                    os.makedirs(SELECTED_BOXES_PATH, exist_ok=True)
                    
                ### Code used for saving the generated input boxes (not essential, can be removed) - start
                # Saving the segmentation image
                Image.fromarray(segmentation_masks * 255).convert("L").save(f"{SELECTED_BOXES_PATH}/{image_name}_segmentation.png")

                # Saving the original input image (unmasked)
                image_arr = inverse_image_transform(images[0, :, :, :]).detach().cpu().numpy()
                Image.fromarray((255 * np.transpose(image_arr, (1, 2, 0))).astype("uint8")).convert("RGB").save(f"{SELECTED_BOXES_PATH}/{image_name}_image.png")
                
                # Saving the input image masked using the sampled anomaly box (only the pixels inside the anomaly box are visible)
                pixels_anomaly_mask = np.repeat(np.expand_dims(dict_masks_target_encoder["pixels_anomaly_mask"], axis=0), 3, axis=0).astype("uint8")
                masked_image = np.transpose((255 * image_arr * pixels_anomaly_mask).astype(np.uint8), (1, 2, 0))
                Image.fromarray(masked_image).convert("RGB").save(f"{SELECTED_BOXES_PATH}/{image_name}_anomaly_pixels_mask_image.png")

                # Saving the input image masked using the complement of the sampled anomaly box (only the pixels outside the anomaly box are visible)
                pixels_anomaly_mask_complement = np.repeat(np.expand_dims(dict_masks_target_encoder["pixels_anomaly_mask_complement"], axis=0), 3, axis=0).astype("uint8")
                complement_masked_image = np.transpose((255 * image_arr * pixels_anomaly_mask_complement).astype(np.uint8), (1, 2, 0))
                Image.fromarray(complement_masked_image).convert("RGB").save(f"{SELECTED_BOXES_PATH}/{image_name}_anomaly_pixels_mask_complement_image.png")
            
                # Saving the input image masked using the sampled normal box (only the pixels inside the normal box are visible)
                pixels_normal_mask = np.repeat(np.expand_dims(dict_masks_target_encoder["pixels_normal_mask"], axis=0), 3, axis=0).astype("uint8")
                masked_image = np.transpose((255 * image_arr * pixels_normal_mask).astype(np.uint8), (1, 2, 0))
                Image.fromarray(masked_image).convert("RGB").save(f"{SELECTED_BOXES_PATH}/{image_name}_normal_pixels_mask_image.png")

                # Saving the input image masked using the complement of the sampled normal box (only the pixels outside the normal box are visible)
                pixels_normal_mask_complement = np.repeat(np.expand_dims(dict_masks_target_encoder["pixels_normal_mask_complement"], axis=0), 3, axis=0).astype("uint8")
                complement_masked_image = np.transpose((255 * image_arr * pixels_normal_mask_complement).astype(np.uint8), (1, 2, 0))
                Image.fromarray(complement_masked_image).convert("RGB").save(f"{SELECTED_BOXES_PATH}/{image_name}_normal_pixels_mask_complement_image.png")
                ### Code used for saving the sampled input boxes - end

            # Convert from the pixels of the anomaly box to the patch indices of the anomaly mask
            anomaly_box = dict_masks_target_encoder["anomaly_box"]  
            anomaly_mask_dict = transform_from_target_box_to_mask(
                image_height=crop_size,
                image_width=crop_size,
                box=anomaly_box,
                patch_size=patch_size
            )
            anomaly_mask_target_encoder = torch.unsqueeze(anomaly_mask_dict["mask"], 0).to(device=device)

            # Convert from the pixels of the normal box to the patch indices of the normal mask
            normal_box = dict_masks_target_encoder["normal_box"]
            normal_mask_dict = transform_from_target_box_to_mask(
                image_height=crop_size,
                image_width=crop_size,
                box=normal_box,
                patch_size=patch_size
                )
            normal_mask_target_encoder = torch.unsqueeze(normal_mask_dict["mask"], 0).to(device=device)
            
            # The target encoder receives as input a list of two tensor masks (anomaly mask and normal mask)
            list_masks_target_encoder = [anomaly_mask_target_encoder, normal_mask_target_encoder]

            encoder_mask = [
                sample_context(
                    enc_mask_scale=enc_mask_scale,
                    aspect_ratio=enc_mask_aspect_ratio,
                    image_height=crop_size,
                    image_width=crop_size,
                    patch_size=patch_size,
                    target_masks=[anomaly_mask_dict["mask"], normal_mask_dict["mask"]],
                    min_keep=enc_min_keep).to(device)
            ]

            if SAVE_IMAGES_WITH_MASKS:
                ### Code used only for saving the images (not essential, can be removed) - start
                # Convert from a tensor of patch indices (mask) to an array of pixels
                encoder_context_mask_arr = transform_from_mask_to_box(
                    image_height=crop_size,
                    image_width=crop_size,
                    mask=encoder_mask[0],
                    patch_size=patch_size,
                )
                # Save the image with the encoder masked context
                encoder_context_mask = np.repeat(np.expand_dims(encoder_context_mask_arr, axis=0), 3, axis=0).astype("uint8")
                masked_image = np.transpose((255 * image_arr * encoder_context_mask).astype(np.uint8), (1, 2, 0))
                Image.fromarray(masked_image).convert("RGB").save(f"{SELECTED_BOXES_PATH}/{image_name}_context_image.png")
                ### Code used only for saving the images - end

            encoder.eval()
            target_encoder.eval()
            predictor.eval()

            with torch.no_grad():
                # --- Target encoder forward prop
                # Forward propagation on the original input image (unmasked, as it was mentioned in the paper)
                target_encoder_emb = target_encoder.forward(images)
                # We do not need this anymore as we're already applying a layer normalization at the end of the target encoder forward function
                # target_encoder_emb = F.layer_norm(target_encoder_emb, (target_encoder_emb.size(-1),))  # normalize over feature-dim
                # retrieve the batch size
                B = len(target_encoder_emb)

                # Retrieve the embeddings only for the mask patches 
                target_encoder_emb = apply_masks(target_encoder_emb, list_masks_target_encoder)
                target_encoder_emb = repeat_interleave_batch(target_encoder_emb, B, repeat=len(encoder_mask)) # repeat = 1 because we are sampling only one context mask

                # --- (Context) Encoder forward prop
                encoder_emb = encoder.forward(images, encoder_mask)

                # --- Predictor forward prop
                predictor_emb = predictor.forward(encoder_emb, encoder_mask, list_masks_target_encoder)

                # --- Compute the metrics
                # Average L2 distance between the predicted and target patch-level representations (embeddings) for the anomaly and normal boxes
                # l2 = torch.linalg.vector_norm(predictor_emb - target_encoder_emb, ord=2, dim=(1, 2))
                l2_per_embedding = torch.linalg.vector_norm(predictor_emb - target_encoder_emb, ord=2, dim=2)
                l2 = torch.mean(input=l2_per_embedding, dim=1, keepdim=False)
                
                # Get the L2 distance between the patch-level representations (embeddings) for the anomaly box (as it was the first element in `list_masks_target_encoder`)
                list_l2_anomaly.append(l2[0].item())
                
                # Get the L2 distance between the patch-level representations (embeddings) for the normal box (as it was the second element in `list_masks_target_encoder`)
                list_l2_normal.append(l2[1].item())

                list_images_names.append(image_name)
                box_sizes.append((target_block_h * patch_size, target_block_w * patch_size))

        print("seed: ", seed)
        print("Mean L2 anomaly mean: ", statistics.mean(list_l2_anomaly))
        print("Mean L2 normal mean: ",statistics.mean(list_l2_normal))

    # Save the results
    df_results = pd.DataFrame({
        "l2_distance_anomaly_box": list_l2_anomaly,
        "l2_distance_normal_box": list_l2_normal,
        "box_size": box_sizes,
        "image_name": list_images_names      
    })
    
    df_results.to_csv(RESULTS_CSV_PATH, index=False)

if __name__ == "__main__":
    main()
