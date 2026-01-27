import torch
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter

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

DATASET_NAME = "BraTS2021_slice"
SPLIT = "valid"

# How many random target boxes to sample per image
NUM_TARGET_BOXES_PER_IMAGE = 30
AGGREGATE_PATCHES_STRATEGY = "mean"
GAUSSIAN_FILTER_BLUR_SIGMA = 4

CONFIG_PATH = f"/{root_directory_path}/{current_dir}/configs/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3.yaml"
CKPT_PATH = f"/{root_directory_path}/{current_dir}/experiments_{DATASET_NAME}/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3/checkpoints/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3-latest.pth.tar"
EVALUATION_DATASET_PATH = f"/{root_directory_path}/BMAD/dataset/{DATASET_NAME}/{SPLIT}"
RANDOM_BOXES_PATH = f"/{root_directory_path}/{current_dir}/experiments_{DATASET_NAME}/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3/results_{SPLIT}/random_boxes_images"

ANOMALY_MAPS_IMAGES_PATH = f"/{root_directory_path}/{current_dir}/experiments_{DATASET_NAME}/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3/results_{SPLIT}/anomaly_maps_images_{AGGREGATE_PATCHES_STRATEGY}_agg_strategy_{NUM_TARGET_BOXES_PER_IMAGE}_boxes_sigma_{GAUSSIAN_FILTER_BLUR_SIGMA}"
ANOMALY_MAPS_ARRAYS_PATH = f"/{root_directory_path}/{current_dir}/experiments_{DATASET_NAME}/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3/results_{SPLIT}/anomaly_maps_arrays_{AGGREGATE_PATCHES_STRATEGY}_agg_strategy_{NUM_TARGET_BOXES_PER_IMAGE}_boxes_sigma_{GAUSSIAN_FILTER_BLUR_SIGMA}"
DISTANCES_RESULTS_CSV_PATH = f"/{root_directory_path}/{current_dir}/experiments_{DATASET_NAME}/enc_vits16_dinov3_pred_vitg16_ijepa_target_vits16_dinov3/results_{SPLIT}/l2_distance_random_box_{NUM_TARGET_BOXES_PER_IMAGE}_seeds.csv"

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
    list_l2_distances_per_image = []
    list_labels = []
    list_seeds = []

    # -- used for building the anomaly map
    num_patches = (crop_size // patch_size) * (crop_size // patch_size)
    image_per_patch_l2_distances = {}
    # --
    
    for box_idx in tqdm(range(NUM_TARGET_BOXES_PER_IMAGE)):
        seed = box_idx
        g.manual_seed(seed)

        target_block_h, target_block_w = mc._sample_block_size(
            generator=g,
            scale=target_block_scale,
            aspect_ratio_scale=target_block_aspect_ratio)
        
        for _, (images, segmentation_masks, labels, image_name) in enumerate(eval_loader):
            images = images.to(device)
            labels = labels.to(device)
            image_name = image_name[0].split('.')[0]
            segmentation_masks = np.squeeze(np.asarray(segmentation_masks))

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

            if SAVE_IMAGES_WITH_MASKS and (seed == 0): # save examples of sampled boxes / images only for the first seed
                if not os.path.exists(RANDOM_BOXES_PATH):
                    os.makedirs(RANDOM_BOXES_PATH, exist_ok=True)
                ### Code used for saving the generated input boxes (not essential, can be removed) - start
                # Saving the original input image (unmasked)
                image_arr = inverse_image_transform(images[0, :, :, :]).detach().cpu().numpy()
                Image.fromarray((255 * np.transpose(image_arr, (1, 2, 0))).astype("uint8")).convert("RGB").save(f"{RANDOM_BOXES_PATH}/{image_name}_image.png")
                Image.fromarray((255 * np.transpose(images[0, :, :, :].detach().cpu().numpy(), (1, 2, 0))).astype("uint8")).convert("RGB").save(f"{RANDOM_BOXES_PATH}/{image_name}_normalized_image.png")
    
                # Saving the input image masked using the sampled random box (only the pixels inside the random box are visible)
                pixels_random_mask = np.repeat(np.expand_dims(dict_mask_target_encoder["pixels_random_mask"], axis=0), 3, axis=0).astype("uint8")
                masked_image = np.transpose((255 * image_arr * pixels_random_mask).astype(np.uint8), (1, 2, 0))
                Image.fromarray(masked_image).convert("RGB").save(f"{RANDOM_BOXES_PATH}/{image_name}_random_pixels_mask_image_seed_{seed}.png")

                # Save the image with the encoder masked context
                encoder_context_mask = np.repeat(np.expand_dims(encoder_context_mask_arr, axis=0), 3, axis=0).astype("uint8")
                masked_image = np.transpose((255 * image_arr * encoder_context_mask).astype(np.uint8), (1, 2, 0))
                Image.fromarray(masked_image).convert("RGB").save(f"{RANDOM_BOXES_PATH}/{image_name}_context_image_seed_{seed}.png")
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
                target_encoder_emb = repeat_interleave_batch(target_encoder_emb, B, repeat=1)

                # --- (Context) Encoder forward prop
                encoder_emb = encoder.forward(images, encoder_mask)

                # --- Predictor forward prop
                predictor_emb = predictor.forward(encoder_emb, encoder_mask, list_masks_target_encoder)

                # --- Compute the metrics
                # Average L2 distance between the predicted and target patch-level representations (embeddings) for the random box
                l2_per_embedding = torch.linalg.vector_norm(predictor_emb - target_encoder_emb, ord=2, dim=2)
                l2_per_image = torch.mean(input=l2_per_embedding, dim=1, keepdim=False)
                list_l2_distances_per_image.append(l2_per_image.item())

                list_image_names.append(image_name)
                list_box_sizes.append((target_block_h * patch_size, target_block_w * patch_size))
                list_labels.append(labels.item())
                list_seeds.append(seed)
                list_segmentation_masks.append(segmentation_masks)

                # Get L2 distances per patch embedding for building an anomaly map
                l2_per_patch = torch.linalg.vector_norm(predictor_emb - target_encoder_emb, ord=2, dim=2)
                for mask in list_masks_target_encoder:
                    mask = torch.squeeze(mask)
                    indices_mask_target = mask.tolist()
                    for idx_tensor, index_patch in enumerate(indices_mask_target):
                        image_per_patch_l2_distances[image_name][index_patch] += [l2_per_patch[0, idx_tensor].item()]

        print("seed: ", seed)
        print("Mean L2 anomaly images mean: ", statistics.mean(
            [l2_dist for l2_dist, label in zip(list_l2_distances_per_image, list_labels) if label == 1]))
        print("Mean L2 normal images mean: ", statistics.mean(
            [l2_dist for l2_dist, label in zip(list_l2_distances_per_image, list_labels) if label == 0]))

    df = pd.DataFrame({
        "image_name": list_image_names,
        "label": list_labels,
        "l2_distance": list_l2_distances_per_image,
        "box_size": list_box_sizes,
        "seed": list_seeds
    })

    if not os.path.exists(ANOMALY_MAPS_ARRAYS_PATH):
        os.makedirs(ANOMALY_MAPS_ARRAYS_PATH, exist_ok=True)
    
    if not os.path.exists(Path(DISTANCES_RESULTS_CSV_PATH).parent):
        os.makedirs(Path(DISTANCES_RESULTS_CSV_PATH).parent, exist_ok=True)

    df.to_csv(DISTANCES_RESULTS_CSV_PATH, index=False)

    anomaly_maps = []
    for image_name in list_image_names:
        anomaly_map = build_anomaly_map_interpolation(
            per_patch_l2_distances_dict=image_per_patch_l2_distances[image_name],
            image_height=crop_size,
            image_width=crop_size,
            patch_size=patch_size,
            agg_strategy=AGGREGATE_PATCHES_STRATEGY
        )
        if GAUSSIAN_FILTER_BLUR_SIGMA != -1:
            anomaly_map = gaussian_filter(anomaly_map, GAUSSIAN_FILTER_BLUR_SIGMA)
        anomaly_maps.append(anomaly_map)
        
    # For visualisation purposes
    # MIN-MAX scaling to convert the pixels in the [0, 1] range so that we can display the anomaly maps
    # Compute the min, max statistics at the dataset level
    if not os.path.exists(ANOMALY_MAPS_IMAGES_PATH):
        os.makedirs(ANOMALY_MAPS_IMAGES_PATH, exist_ok=True)

    max_patch_value = max([np.max(anomaly_map) for anomaly_map in anomaly_maps])
    min_patch_value = min([np.min(anomaly_map) for anomaly_map in anomaly_maps])
    for anomaly_map, segmentation_mask, image_name, label in zip(anomaly_maps, list_segmentation_masks, list_image_names, list_labels):
        # Min-max scaling
        scaled_anomaly_map =  (anomaly_map - min_patch_value) / (max_patch_value - min_patch_value + 1e-8)

        scaled_anomaly_map_path = f"{ANOMALY_MAPS_ARRAYS_PATH}/{image_name}_anomaly_map_label_{label}.npy"
        np.save(file=scaled_anomaly_map_path, arr=scaled_anomaly_map)
        segmentation_mask_path = scaled_anomaly_map_path.replace("_anomaly_map_", "_segmentation_mask_")
        np.save(file=segmentation_mask_path, arr=segmentation_mask)

        Image.fromarray((255 * scaled_anomaly_map).astype("uint8")).convert("L").save(f"{ANOMALY_MAPS_IMAGES_PATH}/{image_name}_anomaly_map_label_{label}.png")
        segmentation_mask = np.squeeze(np.array(segmentation_mask))
        Image.fromarray((segmentation_mask * 255).astype("uint8")).convert("L").save(f"{ANOMALY_MAPS_IMAGES_PATH}/{image_name}_segmentation_label_{label}.png")

        # Convert the anomaly map into a heatmap and superimpose it on the initial image
        scaled_anomaly_map = (scaled_anomaly_map * 255).astype(np.uint8)
        scaled_anomaly_map = cvt2heatmap(scaled_anomaly_map)

        label_subpath = "good" if label == 0 else "Ungood"
        image_path = os.path.join(EVALUATION_DATASET_PATH, label_subpath, "img", image_name + ".png")
        initial_image = cv2.imread(image_path)
        initial_image = cv2.cvtColor(initial_image, cv2.COLOR_BGR2RGB)
        initial_image = cv2.resize(initial_image, dsize=(crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
        initial_image = (initial_image * 255).astype(np.uint8)

        heatmap = cv2.addWeighted(scaled_anomaly_map, 0.4, initial_image, (1 - 0.4), 0)
        cv2.imwrite(f"{ANOMALY_MAPS_IMAGES_PATH}/{image_name}_heatmap.png", heatmap)

if __name__ == "__main__":
    main()

