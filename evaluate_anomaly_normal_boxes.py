import torch
import statistics
import yaml
import numpy as np
import pandas as pd
from PIL import Image

import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToPILImage

from src.utils.utils_evaluation import *
from src.helper import (
    load_checkpoint,
    init_model_dinov3
)
from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch
from src.masks.multiblock import MaskCollator


CONFIG_PATH = "/home/vhondru/istoian/ijepa/configs/brain_mri_dinov3_vits16.yaml"
CKPT_PATH = "/home/vhondru/istoian/ijepa/experiments/dinov3_enc_vits16_pred_vit_tiny_ijepa_target_vits16/checkpoints/jepa-brain-mri-dinov3-vits16-latest.pth.tar"
VALIDATION_DATASET_PATH = "/home/vhondru/istoian/BMAD/dataset/BraTS2021_slice/valid"
SELECTED_BOXES_PATH = "/home/vhondru/istoian/ijepa/experiments/custom_anomaly_normal_boxes"
NUM_SEEDS = 15

def main():
    # Set up the device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Read the configuration file (which was used for generating the checkpoint)
    encoder_model_name = config["meta"]["encoder_model_name"]
    target_encoder_model_name = config["meta"]["target_encoder_model_name"]
    encoder_weights_path = config["meta"]["encoder_weights_path"]
    target_encoder_weights_path = config["meta"]["target_encoder_weights_path"]
    predictor_depth = config["meta"]["predictor_depth"]
    encoder_embed_dim = config["meta"]["encoder_embed_dim"]
    predictor_embed_dim = config["meta"]["predictor_embed_dim"]
    crop_size = config["data"]["crop_size"]
    patch_size = config["mask"]["patch_size"]
    predictor_num_heads = config["meta"]["predictor_num_heads"]
    target_block_scale=config["mask"]["pred_mask_scale"]
    target_block_aspect_ratio=config["mask"]["aspect_ratio"]
    enc_mask_scale = config["mask"]["aspect_ratio"]
    enc_mask_aspect_ratio = config["mask"]["aspect_ratio"]
    min_keep = config["mask"]["min_keep"]

    # Init DinoV3/I-Jepa models
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

    # Load the Dinov3 pre-trained checkpoint
    print(f"Loading checkpoint from: {CKPT_PATH}")
    encoder, predictor, target_encoder, _, _, _ = load_checkpoint(
        r_path=CKPT_PATH,
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        opt=None,
        scaler=None
    )

    # Load the BrainMRI validation dataset
    input_transforms = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor()
    ])
    
    val_dataset = SegmentationWithLabelDataset(
        images_dir=VALIDATION_DATASET_PATH,
        transform=input_transforms,
        mask_transform=input_transforms,
        is_train_dataset=False)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False)
    
    print("Number of validation samples: ", len(val_loader))

    # Sample the size of a target block
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
        
        for it, (images, segmentation_masks, labels, image_name) in enumerate(val_loader):
            if labels[0].item() == 0:
                continue

            images = images.to(device)
            labels = labels.to(device)
            segmentation_masks = np.squeeze(np.array(segmentation_masks))
            
            masks_target_encoder = get_custom_target_anomaly_and_normal_box(
                segmentation_mask=segmentation_masks,
                box_height=target_block_h,
                box_width=target_block_w,
                patch_size=patch_size,
                seed=seed
            )

            Image.fromarray(segmentation_masks * 255).convert("L").save(f"{SELECTED_BOXES_PATH}/{image_name[0].split('.')[0]}_segmentation.png")

            anomaly_box = masks_target_encoder["anomaly_box"]            
            image_arr = images[0, :, :, :].detach().cpu().numpy()
            Image.fromarray((255 * np.transpose(image_arr, (1, 2, 0))).astype("uint8")).convert("RGB").save(f"{SELECTED_BOXES_PATH}/{image_name[0].split('.')[0]}_image.png")
            
            pixels_anomaly_mask = np.repeat(np.expand_dims(masks_target_encoder["pixels_anomaly_mask"], axis=0), 3, axis=0).astype("uint8")
            masked_image = np.transpose((255 * image_arr * pixels_anomaly_mask).astype(np.uint8), (1, 2, 0))
            Image.fromarray(masked_image).convert("RGB").save(f"{SELECTED_BOXES_PATH}/{image_name[0].split('.')[0]}_anomaly_pixels_mask_image.png")

            pixels_anomaly_mask_complement = np.repeat(np.expand_dims(masks_target_encoder["pixels_anomaly_mask_complement"], axis=0), 3, axis=0).astype("uint8")
            complement_masked_image = np.transpose((255 * image_arr * pixels_anomaly_mask_complement).astype(np.uint8), (1, 2, 0))
            Image.fromarray(complement_masked_image).convert("RGB").save(f"{SELECTED_BOXES_PATH}/{image_name[0].split('.')[0]}_anomaly_pixels_mask_complement_image.png")

            normal_box = masks_target_encoder["normal_box"]
            pixels_normal_mask = np.repeat(np.expand_dims(masks_target_encoder["pixels_normal_mask"], axis=0), 3, axis=0).astype("uint8")
            masked_image = np.transpose((255 * image_arr * pixels_normal_mask).astype(np.uint8), (1, 2, 0))
            Image.fromarray(masked_image).convert("RGB").save(f"{SELECTED_BOXES_PATH}/{image_name[0].split('.')[0]}_normal_pixels_mask_image.png")

            pixels_normal_mask_complement = np.repeat(np.expand_dims(masks_target_encoder["pixels_normal_mask_complement"], axis=0), 3, axis=0).astype("uint8")
            complement_masked_image = np.transpose((255 * image_arr * pixels_normal_mask_complement).astype(np.uint8), (1, 2, 0))
            Image.fromarray(complement_masked_image).convert("RGB").save(f"{SELECTED_BOXES_PATH}/{image_name[0].split('.')[0]}_normal_pixels_mask_complement_image.png")

            anomaly_mask_dict = transform_from_target_box_to_mask(
                image_height=crop_size,
                image_width=crop_size,
                box=anomaly_box,
                patch_size=patch_size
            )

            anomaly_mask_target_encoder = torch.unsqueeze(anomaly_mask_dict["mask"], 0).to(device=device)

            normal_mask_dict = transform_from_target_box_to_mask(
                image_height=crop_size,
                image_width=crop_size,
                box=normal_box,
                patch_size=patch_size
                )
            normal_mask_target_encoder = torch.unsqueeze(normal_mask_dict["mask"], 0).to(device=device)
            list_masks_target_encoder = [anomaly_mask_target_encoder, normal_mask_target_encoder]

            encoder_mask = [
                sample_context(
                    enc_mask_scale=enc_mask_scale,
                    aspect_ratio=enc_mask_aspect_ratio,
                    image_height=crop_size,
                    image_width=crop_size,
                    patch_size=patch_size,
                    target_masks=[anomaly_mask_dict["mask"], normal_mask_dict["mask"]],
                    min_keep=min_keep).to(device)
            ]

            # Save the image using the encoder context mask

            encoder_context_mask_arr = transform_from_mask_to_box(
                image_height=crop_size,
                image_width=crop_size,
                mask=encoder_mask[0],
                patch_size=patch_size,
            )
            encoder_context_mask = np.repeat(np.expand_dims(encoder_context_mask_arr, axis=0), 3, axis=0).astype("uint8")
            masked_image = np.transpose((255 * image_arr * encoder_context_mask).astype(np.uint8), (1, 2, 0))
            Image.fromarray(masked_image).convert("RGB").save(f"{SELECTED_BOXES_PATH}/{image_name[0].split('.')[0]}_context_image.png")

            encoder.eval()
            target_encoder.eval()
            predictor.eval()
            with torch.no_grad():
                target_encoder_emb = target_encoder.forward(images)
                target_encoder_emb = F.layer_norm(target_encoder_emb, (target_encoder_emb.size(-1),))  # normalize over feature-dim
                B = len(target_encoder_emb)
                
                target_encoder_emb = apply_masks(target_encoder_emb, list_masks_target_encoder)
                target_encoder_emb = repeat_interleave_batch(target_encoder_emb, B, repeat=1)

                encoder_emb = encoder.forward(images, encoder_mask)
                predictor_emb = predictor.forward(encoder_emb, encoder_mask, list_masks_target_encoder)

                l2 = torch.linalg.vector_norm(predictor_emb - target_encoder_emb, dim=(1, 2))
                list_l2_anomaly.append(l2[0].item())
                list_l2_normal.append(l2[1].item())
                list_images_names.append(image_name[0].split('.')[0])
                box_sizes.append((target_block_h * patch_size, target_block_w * patch_size))

        print("seed: ", seed)
        print("list_l2_anomaly mean: ", statistics.mean(list_l2_anomaly))
        print("list_l2_normal_mean: ",statistics.mean(list_l2_normal))

    df_results = pd.DataFrame({
        "l2_distance_anomaly_box": list_l2_anomaly,
        "l2_distance_normal_box": list_l2_normal,
        "box_size": box_sizes,
        "image_name": list_images_names      
    })
    
    df_results.to_csv("./experiments/l2_distance_anomaly_vs_normal_box_validation_multiple_seeds.csv", index=False)

if __name__ == "__main__":
    main()


