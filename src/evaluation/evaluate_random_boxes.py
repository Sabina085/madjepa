import torch
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch.nn.functional as F
from torchvision import transforms

from src.utils.utils_evaluation import *
from src.helper import (
    load_checkpoint,
    init_model_dinov3
)
from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch
from src.masks.multiblock import MaskCollator


CONFIG_PATH = "/home/vhondru/istoian/ijepa-anomaly-detection/configs/brain_mri_dinov3_vits16.yaml"
CKPT_PATH = "/home/vhondru/istoian/ijepa-anomaly-detection/experiments/enc_vits16_dinov3_pred_vith14_ijepa_target_vits16_dinov3/checkpoints/jepa-brain-mri-dinov3-vits16-latest.pth.tar"
VALIDATION_DATASET_PATH = "/home/vhondru/istoian/BMAD/dataset/BraTS2021_slice/valid"
RANDOM_BOXES_PATH = "/home/vhondru/istoian/ijepa-anomaly-detection/experiments/enc_vits16_dinov3_pred_vith14_ijepa_target_vits16_dinov3/random_boxes"
RESULTS_CSV_PATH = "/home/vhondru/istoian/ijepa-anomaly-detection/experiments/enc_vits16_dinov3_pred_vith14_ijepa_target_vits16_dinov3/l2_distance_random_box_multiple_seeds.csv"
NUM_TARGET_BOXES_PER_IMAGE = 30

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
    print("Config file: ", config)
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
        transforms.Resize((crop_size, crop_size)), # or transforms.CenterCrop(crop_size)
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

    list_box_sizes = []
    list_image_names = []
    list_l2_distances_per_image = []
    list_labels = []
    list_seeds = []
    num_patches = (crop_size // patch_size) * (crop_size // patch_size)
    per_patch_l2_distances = {patch_idx: [] for patch_idx in range(num_patches)}
    for box_idx in tqdm(range(NUM_TARGET_BOXES_PER_IMAGE)):
        seed = box_idx
        g = torch.Generator()
        g.manual_seed(seed)

        target_block_h, target_block_w = mc._sample_block_size(
            generator=g,
            scale=target_block_scale,
            aspect_ratio_scale=target_block_aspect_ratio)
        
        for _, (images, segmentation_masks, labels, image_name) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            segmentation_masks = np.squeeze(np.array(segmentation_masks))

            dict_mask_target_encoder = get_random_box(
                image_height=crop_size,
                image_width=crop_size,
                box_height=target_block_h,
                box_width=target_block_w,
                patch_size=patch_size,
                seed=seed
            )
            random_box = dict_mask_target_encoder["random_box"]

            image_arr = images[0, :, :, :].detach().cpu().numpy()
            Image.fromarray((255 * np.transpose(image_arr, (1, 2, 0))).astype("uint8")).convert("RGB").save(f"{RANDOM_BOXES_PATH}/{image_name[0].split('.')[0]}_image.png")
 
            pixels_anomaly_mask = np.repeat(np.expand_dims(dict_mask_target_encoder["pixels_random_mask"], axis=0), 3, axis=0).astype("uint8")
            masked_image = np.transpose((255 * image_arr * pixels_anomaly_mask).astype(np.uint8), (1, 2, 0))
            Image.fromarray(masked_image).convert("RGB").save(f"{RANDOM_BOXES_PATH}/{image_name[0].split('.')[0]}_random_pixels_mask_image_{seed}.png")

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
                min_keep=min_keep
            ).to(device)]

            # Save the image using the encoder context mask

            encoder_context_mask_arr = transform_from_mask_to_box(
                image_height=crop_size,
                image_width=crop_size,
                mask=encoder_mask[0],
                patch_size=patch_size,
            )
            encoder_context_mask = np.repeat(np.expand_dims(encoder_context_mask_arr, axis=0), 3, axis=0).astype("uint8")
            masked_image = np.transpose((255 * image_arr * encoder_context_mask).astype(np.uint8), (1, 2, 0))
            Image.fromarray(masked_image).convert("RGB").save(f"{RANDOM_BOXES_PATH}/{image_name[0].split('.')[0]}_context_image_{seed}.png")
        
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

                l2_per_image = torch.linalg.vector_norm(predictor_emb - target_encoder_emb, dim=(1, 2))
                
                list_image_names.append(image_name[0].split('.')[0])
                list_box_sizes.append((target_block_h * patch_size, target_block_w * patch_size))
                list_l2_distances_per_image.append(l2_per_image.item())
                list_labels.append(labels.item())
                list_seeds.append(seed)

                # Get L2 distances per patch embedding for building an anomaly map
                l2_per_patch = torch.linalg.vector_norm(predictor_emb - target_encoder_emb, dim=2)
                for mask in list_masks_target_encoder:
                    indices_mask_target = mask.tolist()
                    for idx_tensor, index_patch in enumerate(indices_mask_target):
                        per_patch_l2_distances[index_patch] += [l2_per_patch[idx_tensor].item()]

            
    df = pd.DataFrame({
        "image_name": list_image_names,
        "label": list_labels,
        "l2_distance": list_l2_distances_per_image,
        "box_size": list_box_sizes,
        "seed": list_seeds
    })
    df.to_csv(RESULTS_CSV_PATH, index=False)


if __name__ == "__main__":
    main()

