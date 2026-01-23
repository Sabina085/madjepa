from typing import Dict, Tuple, List
from collections import Counter
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import statistics
import glob
import cv2
import torch
import random
from src.masks.multiblock import MaskCollator

label2idx = {"good": 0, "Ungood": 1}
idx2label = {0: "good", 1: "Ungood"}


def get_custom_target_anomaly_and_normal_box(
        segmentation_mask: np.ndarray,
        box_height: int,
        box_width: int,
        patch_size: int,
        seed: int
) -> Dict[str, List[Tuple[int, int]] | np.ndarray]:
    image_height, image_width = segmentation_mask.shape

    box_height *= patch_size
    box_width *= patch_size
    anomaly_count_pixels = dict()

    for h in range(0, image_height - box_height, patch_size):
        for w in range(0, image_width - box_width, patch_size):
            crt_box = segmentation_mask[h: h + box_height, w: w + box_width]
            counter_pixels = Counter(crt_box.flatten())
            num_anomaly_pixels = (box_height * box_width) - counter_pixels[0]
            anomaly_count_pixels[num_anomaly_pixels] = anomaly_count_pixels.get(
                num_anomaly_pixels, []) + [[(h, w), (h + box_height, w + box_width)]]

    sorted_dict = sorted(anomaly_count_pixels.items(), key=lambda x: x[0], reverse=True)

    np.random.seed(seed)
    anomaly_box = sorted_dict[0][1][np.random.randint(0, len(sorted_dict[0][1]))]
    normal_box = sorted_dict[-1][1][np.random.randint(0, len(sorted_dict[-1][1]))]

    pixels_anomaly_mask = np.zeros((image_height, image_width))
    pixels_anomaly_mask[anomaly_box[0][0]: anomaly_box[1][0], anomaly_box[0][1]: anomaly_box[1][1]] = 1
    pixels_anomaly_mask_complement = np.ones((image_height, image_width))
    pixels_anomaly_mask_complement[anomaly_box[0][0]: anomaly_box[1][0], anomaly_box[0][1]: anomaly_box[1][1]] = 0

    pixels_normal_mask = np.zeros((image_height, image_width))
    pixels_normal_mask[normal_box[0][0]: normal_box[1][0], normal_box[0][1]: normal_box[1][1]] = 1
    pixels_normal_mask_complement = np.ones((image_height, image_width))
    pixels_normal_mask_complement[normal_box[0][0]: normal_box[1][0], normal_box[0][1]: normal_box[1][1]] = 0

    return {
        "anomaly_box": anomaly_box,
        "normal_box": normal_box,
        "pixels_anomaly_mask": pixels_anomaly_mask,
        "pixels_anomaly_mask_complement": pixels_anomaly_mask_complement,
        "pixels_normal_mask": pixels_normal_mask,
        "pixels_normal_mask_complement": pixels_normal_mask_complement,
    }

def get_random_box(
        image_height: int,
        image_width: int,
        box_height: int,
        box_width: int,
        patch_size: int,
        seed: int
) -> Dict[str, List[Tuple[int, int]] | np.ndarray]:
    box_height *= patch_size
    box_width *= patch_size

    np.random.seed(seed=seed)
    random.seed(seed=seed)
    height_sampling_locations = [
        x for x in range(0, image_height - box_height + 1, patch_size)
    ]
    width_sampling_locations = [
        x for x in range(0, image_width - box_width + 1, patch_size)
    ]
    top = random.sample(population=height_sampling_locations, k=1)[0]
    left = random.sample(population=width_sampling_locations, k=1)[0]
    random_box = [(top, left), (top + box_height, left + box_width)]

    pixels_random_mask = np.zeros(shape=(image_height, image_width))
    pixels_random_mask[random_box[0][0]: random_box[1][0], random_box[0][1]: random_box[1][1]] = 1

    pixels_random_mask_complement = np.ones(shape=(image_height, image_width))
    pixels_random_mask_complement[random_box[0][0]: random_box[1][0], random_box[0][1]: random_box[1][1]] = 0

    return {
        "random_box": random_box,
        "pixels_random_mask": pixels_random_mask,
        "pixels_random_mask_complement": pixels_random_mask_complement
    }

def transform_from_target_box_to_mask(
        image_height: int,
        image_width: int,
        box: List[Tuple[int, int]],
        patch_size: int,
        ) -> Dict[str, torch.tensor]:
    
    image_height //= patch_size
    image_width //= patch_size

    start_height = box[0][0] // patch_size
    start_width = box[0][1] // patch_size
    end_height = box[1][0] // patch_size
    end_width = box[1][1] // patch_size

    mask = torch.zeros((image_height, image_width), dtype=torch.int32)
    mask[start_height: end_height, start_width: end_width] = 1
    mask = torch.nonzero(mask.flatten())
    mask = mask.squeeze()
    
    mask_complement = torch.ones((image_height, image_width), dtype=torch.int32)
    mask_complement[start_height: end_height, start_width: end_width] = 0
    
    return {
        "mask": mask,
        "mask_complement": mask_complement
    }

def transform_from_mask_to_box(
        image_height: int,
        image_width: int,
        mask: torch.tensor,
        patch_size: int
) -> np.ndarray:
    num_patches = (image_height // patch_size) * (image_width // patch_size)
    arr_mask = np.zeros(shape=(image_height, image_width), dtype=np.uint8)
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size

    for patch_idx in range(num_patches):
        if torch.isin(patch_idx, mask).item():
            row = patch_idx // num_patches_width
            col = patch_idx % num_patches_width
            arr_mask[
                row * patch_size: (row + 1) * patch_size,
                col * patch_size: (col + 1) * patch_size 
            ] = 1

    return arr_mask


def sample_context(
        enc_mask_scale: Tuple[float, float],
        aspect_ratio: Tuple[float, float],
        image_height: int,
        image_width: int,
        patch_size: int,     
        target_masks: List[torch.Tensor],
        min_keep: int 
        ):

    it = 1
    mc = MaskCollator(
        input_size=(image_height, image_width),
        patch_size=patch_size
    )
    
    g = torch.Generator()
    image_height //= patch_size
    image_width //= patch_size

    while it:
        g.manual_seed(it)

        context_height, context_width = mc._sample_block_size(
            generator=g,
            scale=enc_mask_scale,
            aspect_ratio_scale=aspect_ratio
        )
        print("context_height: ", context_height)
        print("context_width: ", context_width)

        np.random.seed(it)
        start_height = np.random.randint(low=0, high=image_height - context_height)
        end_height = start_height + context_height
        start_width = np.random.randint(low=0, high=image_width - context_width)
        end_width = start_width + context_width

        encoder_mask = torch.zeros((image_height, image_width), dtype=torch.int32)
        encoder_mask[start_height: end_height, start_width: end_width] = 1
        encoder_mask = torch.nonzero(encoder_mask.flatten())
        encoder_mask = encoder_mask.squeeze()

        print("encoder_mask.shape before: ", encoder_mask.shape)
        # Remove from the context the patches contained in the target boxes
        for target_mask in target_masks:
            # encoder_mask = torch.tensor(list(set(encoder_mask.tolist()) - set(target_mask.tolist())))
            # cleaner version
            encoder_mask = encoder_mask[~torch.isin(encoder_mask, target_mask)]

        print("encoder_mask.shape after: ", encoder_mask.shape)
        if encoder_mask.shape[0] >= min_keep:
            return torch.unsqueeze(encoder_mask, 0)
        else:
            it += 1

def zscore_sigmoid(anomaly_map, eps=1e-8):
    """
    Z-score + sigmoid normalization for a 2D NumPy anomaly map.

    Args:
        anomaly_map (np.ndarray): 2D array of real-valued anomaly scores
        eps (float): numerical stability constant

    Returns:
        np.ndarray: anomaly map normalized to [0, 1]
    """

    anomaly_map = anomaly_map.astype(np.float32)
    mean = anomaly_map.mean()
    std = anomaly_map.std()

    z = (anomaly_map - mean) / (std + eps)

    return 1.0 / (1.0 + np.exp(-z))

def build_anomaly_map(
        per_patch_l2_distances_dict: Dict[str, List[float]],
        image_height: int,
        image_width: int,
        patch_size: int,
        agg_strategy: str= "mean"
    ):
    num_patches_width = image_width // patch_size

    anomaly_map = np.zeros((image_height, image_width))
    for patch_idx, l2_distances in per_patch_l2_distances_dict.items():
        if len(l2_distances) > 0:
            if agg_strategy == "mean":
                aggregated_distance_per_patch = statistics.mean(l2_distances)
            else:
                aggregated_distance_per_patch = max(l2_distances)

            row = patch_idx // num_patches_width
            col = patch_idx % num_patches_width
            
            anomaly_map[
                row * patch_size: (row + 1) * patch_size,
                col * patch_size: (col + 1) * patch_size] = aggregated_distance_per_patch
    
    return anomaly_map

def build_anomaly_map_interpolation(
        per_patch_l2_distances_dict: Dict[str, List[float]],
        image_height: int,
        image_width: int,
        patch_size: int,
        agg_strategy: str= "mean"
    ):
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size

    patches_anomaly_map = np.zeros((num_patches_height, num_patches_width))
    for patch_idx, l2_distances in per_patch_l2_distances_dict.items():
        if len(l2_distances) > 0:
            if agg_strategy == "mean":
                aggregated_distance_per_patch = statistics.mean(l2_distances)
            else:
                aggregated_distance_per_patch = max(l2_distances)

            row = patch_idx // num_patches_width
            col = patch_idx % num_patches_width
            patches_anomaly_map[row, col] = aggregated_distance_per_patch

    anomaly_map = cv2.resize(
        patches_anomaly_map, dsize=(image_height, image_width), interpolation=cv2.INTER_LINEAR)

    return anomaly_map


class SegmentationWithLabelDataset(Dataset):
    def __init__(self, images_dir, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.mask_transform = mask_transform

        # images
        self.all_paths = sorted(glob.glob(os.path.join(images_dir, "**/*.png"), recursive=True))
        self.images = [img_path for img_path in self.all_paths if 'label' not in img_path]
        self.image_names = [img_path.split("/")[-1] for img_path in self.images]

        # segmentation masks
        self.masks = [img_path.replace("img", "label") for img_path in self.images]
            
        # labels     
        self.labels_names = [img_path.split("/")[8] for img_path in self.images]
        self.labels = [label2idx[label_name] for label_name in self.labels_names]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load image
        img_path = self.images[idx]
        img_name = self.image_names[idx]
        image = Image.open(img_path).convert("RGB")

        # load mask
        mask_path = self.masks[idx]
        mask = Image.open(mask_path).convert("L")
            
        # load label
        label = self.labels[idx]

        # apply transformations
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            # Preprocessing the segmentation masks: from BMAD codebase: https://github.com/DorisBao/BMAD/blob/main/UTRAD/data.py#L130
            mask[mask >= 0.1] = 1
            mask[mask < 0.1] = 0

        return image, mask, label, img_name
    

class LabelDataset(Dataset):
    # This is a simpler version of the dataset. It does not include segmentation masks, only the image and the label.
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        # images
        self.all_paths = sorted(glob.glob(os.path.join(images_dir, "**/*.png"), recursive=True))
        self.images = [img_path for img_path in self.all_paths if 'label' not in img_path]
        self.image_names = [img_path.split("/")[-1] for img_path in self.images]
            
        # labels     
        self.labels_names = [img_path.split("/")[8] for img_path in self.images]
        self.labels = [label2idx[label_name] for label_name in self.labels_names]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load image
        img_path = self.images[idx]
        img_name = self.image_names[idx]
        image = Image.open(img_path).convert("RGB")
            
        # load label
        label = self.labels[idx]

        # apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label, img_name

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return heatmap
