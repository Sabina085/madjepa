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
RANDOM_BOXES_PATH = f"/{root_directory_path}/{current_dir}/experiments/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3/random_boxes"
ANOMALY_MAPS_PATH = f"/{root_directory_path}/{current_dir}/experiments/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3/anomaly_maps"
RESULTS_CSV_PATH = f"/{root_directory_path}/{current_dir}/experiments/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3/l2_distance_random_box_multiple_seeds.csv"

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
    # Read the configuration file (which was used for generating the checkpoint)
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
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1129, 0.1240, 0.1445), std=(0.0392, 0.0495, 0.0612)),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor()
    ])
    
    val_dataset = SegmentationWithLabelDataset(
        images_dir=VALIDATION_DATASET_PATH,
        transform=image_transform,
        mask_transform=mask_transform)

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
    list_segmentation_masks = []
    list_l2_distances_per_image = []
    list_labels = []
    list_seeds = []

    # -- useful for building the anomaly map
    num_patches = (crop_size // patch_size) * (crop_size // patch_size)
    image_per_patch_l2_distances = {}
    # --
    
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
            image_name = image_name[0].split('.')[0]
            segmentation_masks = np.squeeze(np.array(segmentation_masks))
            if box_idx == 0:
                image_per_patch_l2_distances[image_name] = {patch_idx: [] for patch_idx in range(num_patches)}

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
            Image.fromarray((255 * np.transpose(image_arr, (1, 2, 0))).astype("uint8")).convert("RGB").save(f"{RANDOM_BOXES_PATH}/{image_name}_image.png")
 
            pixels_anomaly_mask = np.repeat(np.expand_dims(dict_mask_target_encoder["pixels_random_mask"], axis=0), 3, axis=0).astype("uint8")
            masked_image = np.transpose((255 * image_arr * pixels_anomaly_mask).astype(np.uint8), (1, 2, 0))
            Image.fromarray(masked_image).convert("RGB").save(f"{RANDOM_BOXES_PATH}/{image_name}_random_pixels_mask_image_{seed}.png")

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

            # Save the image using the encoder context mask

            encoder_context_mask_arr = transform_from_mask_to_box(
                image_height=crop_size,
                image_width=crop_size,
                mask=encoder_mask[0],
                patch_size=patch_size,
            )
            encoder_context_mask = np.repeat(np.expand_dims(encoder_context_mask_arr, axis=0), 3, axis=0).astype("uint8")
            masked_image = np.transpose((255 * image_arr * encoder_context_mask).astype(np.uint8), (1, 2, 0))
            Image.fromarray(masked_image).convert("RGB").save(f"{RANDOM_BOXES_PATH}/{image_name}_context_image_{seed}.png")
        
            encoder.eval()
            target_encoder.eval()
            predictor.eval()
            with torch.no_grad():
                target_encoder_emb = target_encoder.forward(images)
                # We do not need this anymore as we're already applying a layer normalization at the end of the target encoder forward function
                # target_encoder_emb = F.layer_norm(target_encoder_emb, (target_encoder_emb.size(-1),))  # normalize over feature-dim
                B = len(target_encoder_emb)
                
                target_encoder_emb = apply_masks(target_encoder_emb, list_masks_target_encoder)
                target_encoder_emb = repeat_interleave_batch(target_encoder_emb, B, repeat=1)

                encoder_emb = encoder.forward(images, encoder_mask)
                predictor_emb = predictor.forward(encoder_emb, encoder_mask, list_masks_target_encoder)

                l2_per_image = torch.linalg.vector_norm(predictor_emb - target_encoder_emb, dim=(1, 2))
                
                list_image_names.append(image_name)
                list_box_sizes.append((target_block_h * patch_size, target_block_w * patch_size))
                list_l2_distances_per_image.append(l2_per_image.item())
                list_labels.append(labels.item())
                list_seeds.append(seed)
                list_segmentation_masks.append(segmentation_masks)

                # Get L2 distances per patch embedding for building an anomaly map
                l2_per_patch = torch.linalg.vector_norm(predictor_emb - target_encoder_emb, dim=2)
                for mask in list_masks_target_encoder:
                    mask = torch.squeeze(mask)
                    indices_mask_target = mask.tolist()
                    for idx_tensor, index_patch in enumerate(indices_mask_target):
                        image_per_patch_l2_distances[image_name][index_patch] += [l2_per_patch[0, idx_tensor].item()]
                
    
    df = pd.DataFrame({
        "image_name": list_image_names,
        "label": list_labels,
        "l2_distance": list_l2_distances_per_image,
        "box_size": list_box_sizes,
        "seed": list_seeds
    })
    df.to_csv(RESULTS_CSV_PATH, index=False)

    for image_name, segmentation_mask in zip(list_image_names, list_segmentation_masks):
        anomaly_map = build_anomaly_map(
            per_patch_l2_distances_dict=image_per_patch_l2_distances[image_name],
            image_height=crop_size,
            image_width=crop_size,
            patch_size=patch_size
        )
        Image.fromarray((255 * anomaly_map).astype("uint8")).convert("L").save(f"{ANOMALY_MAPS_PATH}/{image_name}_anomaly_map.png")
        segmentation_mask = np.squeeze(np.array(segmentation_mask))
        Image.fromarray((segmentation_mask * 255).astype("uint8")).convert("L").save(f"{ANOMALY_MAPS_PATH}/{image_name}_segmentation.png")

if __name__ == "__main__":
    main()

