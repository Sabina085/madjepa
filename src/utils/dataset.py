import glob
import os
import numpy as np
import uuid
import json

from PIL import Image
from torch.utils.data import Dataset

from src.utils.preprocessing import *

label2idx = {"good": 0, "ungood": 1}
idx2label = {0: "good", 1: "ungood"}

class SegmentationWithLabelDataset(Dataset):
    def __init__(self, images_dir, dataset_name, count_border_pixels, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.dataset_name = dataset_name
        self.count_border_pixels = count_border_pixels

        # images
        self.all_paths = sorted(glob.glob(os.path.join(images_dir, "**/*.png"), recursive=True))
        self.images = [img_path for img_path in self.all_paths if 'label' not in img_path]
        self.image_names = [img_path.split("/")[-1] for img_path in self.images]

        # segmentation masks
        self.masks = [img_path.replace("img", "label") for img_path in self.images]
            
        # labels     
        self.labels_names = [img_path.split("/")[-3] for img_path in self.images]
        self.labels = [label2idx[label_name] for label_name in self.labels_names]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load image
        img_path = self.images[idx]
        img_name = self.image_names[idx]
        original_image = Image.open(img_path).convert("RGB")

        # load mask
        mask_path = self.masks[idx]
        mask = Image.open(mask_path).convert("L")
            
        # load label
        label = self.labels[idx]

        # crop the background border from the image
        image_arr = np.array(original_image)

        if self.dataset_name == "RESC":
            image_arr = remove_noise(image_arr)

        borders_dict = get_image_background_borders(image_arr=image_arr, count_border_pixels=self.count_border_pixels)
        image = original_image.crop(box=(borders_dict["left"], borders_dict["top"], borders_dict["right"], borders_dict["bottom"]))
        
        # apply transformations
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            # Preprocessing the segmentation masks: from BMAD codebase: https://github.com/DorisBao/BMAD/blob/main/UTRAD/data.py#L130
            mask[mask >= 0.1] = 1
            mask[mask < 0.1] = 0

        return image, mask, label, img_name, borders_dict
    

class LabelDataset(Dataset):
    # This is a simpler version of the dataset. It does not include segmentation masks, only the image and the label.
    def __init__(self, images_dir, dataset_name, count_border_pixels, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.dataset_name = dataset_name
        self.count_border_pixels = count_border_pixels

        # images
        self.all_paths = sorted(glob.glob(os.path.join(images_dir, "**/*.png"), recursive=True))
        self.images = [img_path for img_path in self.all_paths if 'label' not in img_path]
        self.image_names = [img_path.split("/")[-1] for img_path in self.images]
            
        # labels     
        self.labels_names = [img_path.split("/")[-3] for img_path in self.images]
        self.labels = [label2idx[label_name] for label_name in self.labels_names]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load image
        img_path = self.images[idx]
        img_name = self.image_names[idx]
        original_image = Image.open(img_path).convert("RGB")
            
        # load label
        label = self.labels[idx]

        # crop the background border from the image
        image_arr = np.array(original_image)

        if self.dataset_name == "RESC":
            image_arr = remove_noise(image_arr)

        borders_dict = get_image_background_borders(image_arr=image_arr, count_border_pixels=self.count_border_pixels)
        image = original_image.crop(box=(borders_dict["left"], borders_dict["top"], borders_dict["right"], borders_dict["bottom"]))
        
        # apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label, img_name, borders_dict

class ImageDataset(Dataset):
     # This is a simpler version of the dataset. It does not include segmentation masks or classification labels, only the images.
    def __init__(self, images_dir, dataset_name, count_border_pixels, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.dataset_name = dataset_name
        self.count_border_pixels = count_border_pixels

        # images
        self.all_paths = sorted(glob.glob(os.path.join(images_dir, "**/*.png"), recursive=True))
        self.images = [img_path for img_path in self.all_paths if 'label' not in img_path]
        self.image_names = [img_path.split("/")[-1] for img_path in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load image
        img_path = self.images[idx]
        img_name = self.image_names[idx]
        original_image = Image.open(img_path).convert("RGB")

        # crop the background border from the image
        image_arr = np.array(original_image)

        if self.dataset_name == "RESC":
            image_arr = remove_noise(image_arr)

        borders_dict = get_image_background_borders(image_arr=image_arr, count_border_pixels=self.count_border_pixels)
        image = original_image.crop(box=(borders_dict["left"], borders_dict["top"], borders_dict["right"], borders_dict["bottom"]))
        
        # apply transformations
        if self.transform:
            image = self.transform(image)

        return image, img_name, borders_dict
    
