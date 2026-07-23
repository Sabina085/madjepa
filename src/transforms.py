# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from src.utils.preprocessing import *

from PIL import Image
import numpy as np
import uuid
import json

import torchvision.transforms as transforms

logger = getLogger()

def make_transforms(
    dataset_name,
    crop_size=224,
    crop_scale=(0.8, 1.0),
    # ImageNet normalization statistics
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225)),
    count_border_pixels=2
):
    logger.info('Applying imagenet stats for normalization')

    transform_list = [CropBackgroundBordersTransform(dataset_name, count_border_pixels)]
    # transform_list += [transforms.Resize((crop_size, crop_size), Image.BICUBIC)]
    transform_list += [transforms.RandomResizedCrop(
        crop_size, 
        scale=crop_scale, 
        interpolation=transforms.InterpolationMode.BICUBIC)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(mean=normalization[0], std=normalization[1])]

    transform = transforms.Compose(transform_list)

    return transform

class CropBackgroundBordersTransform:
    def __init__(self, dataset_name, count_border_pixels=2):
        self.dataset_name=dataset_name
        self.count_border_pixels = count_border_pixels

    def __call__(self, image):
        # image is PIL Image
        # convert to numpy array
        image_arr = np.array(image)
        if self.dataset_name in ["RESC"]:
            threshold_map = get_binary_threshold_map(image_arr)
        else:
            threshold_map = image_arr.copy()

        borders_dict = get_image_background_borders(
            image_arr=threshold_map,
            count_border_pixels=self.count_border_pixels
        )
        
        cropped_img = image.crop(
            box=(
                borders_dict["left"],
                borders_dict["top"], 
                borders_dict["right"], 
                borders_dict["bottom"]
            ))
        
        if self.dataset_name in ["hist_DIY"]:
            background_white_image = convert_background_to_white(
                image_original=np.asarray(cropped_img), 
                closing_kernel_size=7,
                dataset=self.dataset_name.lower())
            cropped_img = Image.fromarray(background_white_image)

        return cropped_img
