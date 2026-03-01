# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from src.utils.preprocessing import get_image_background_borders

from PIL import Image
import numpy as np
import uuid

import torchvision.transforms as transforms

logger = getLogger()

def make_transforms(
    crop_size=224,
    crop_scale=(0.8, 1.0),
    # ImageNet normalization statistics
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    logger.info('Applying imagenet stats for normalization')

    transform_list = [CropBackgroundBordersTransform()]
    # transform_list += [transforms.Resize((crop_size, crop_size), Image.BICUBIC)]
    transform_list += [transforms.RandomResizedCrop(
        crop_size, scale=crop_scale, 
        interpolation=transforms.InterpolationMode.BICUBIC)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(mean=normalization[0], std=normalization[1])]

    transform = transforms.Compose(transform_list)

    return transform

class CropBackgroundBordersTransform:
    def __call__(self, image, count_border_pixels=2):
        # image is PIL Image
        # convert to numpy array
        image_arr = np.array(image)
        borders_dict = get_image_background_borders(image_arr=image_arr, count_border_pixels=count_border_pixels)
        cropped_img = image.crop(box=(borders_dict["left"], borders_dict["top"], borders_dict["right"], borders_dict["bottom"]))
        # uid = uuid.uuid4()
        # image.save(f"Test_cropped_images/{uid}_original.jpg")
        # cropped_img.save(f"Test_cropped_images/{uid}_cropped.jpg")

        return cropped_img
