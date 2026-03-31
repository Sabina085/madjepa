import glob
import os
import numpy as np
from pathlib import Path
from PIL import Image

IMAGES_FOLDER_PATH = "/media/vhondru/hdd/istoian/BMAD/dataset/RESC/test/good/img"
SEGMENTATION_MASKS_FOLDER_PATH = IMAGES_FOLDER_PATH.replace("img", "label")

os.makedirs(SEGMENTATION_MASKS_FOLDER_PATH, exist_ok=True)

image_paths = glob.glob(os.path.join(IMAGES_FOLDER_PATH, "*"))
for image_path in image_paths:
    image_name = Path(image_path).name
    segmentation_path = os.path.join(SEGMENTATION_MASKS_FOLDER_PATH, image_name)

    image = Image.open(image_path)
    width, height = image.size

    segmentation_arr = np.zeros((height, width), dtype=np.uint8)
    segmentation_image = Image.fromarray(segmentation_arr)
    print(f"Saving image to: {segmentation_path} ...")
    segmentation_image.save(segmentation_path)

