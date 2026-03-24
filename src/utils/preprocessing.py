import cv2
import numpy as np
from typing import Dict


def get_image_background_borders(image_arr: np.ndarray, count_border_pixels=2) -> Dict[str, int]:
    num_rows, num_cols = image_arr.shape[:2]

    # Create mask of non-background pixels
    if image_arr.ndim == 3:
        mask = np.any(image_arr > 0, axis=2)
    else:
        mask = image_arr > 0

    coords = np.argwhere(mask)

    if len(coords) == 0:
        # image is fully background
        return {
            "left": 0,
            "top": 0,
            "right": num_cols - 1,
            "bottom": num_rows - 1
        }

    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)

    # Add padding safely
    min_row = max(min_row - count_border_pixels, 0)
    max_row = min(max_row + count_border_pixels, num_rows - 1)

    min_col = max(min_col - count_border_pixels, 0)
    max_col = min(max_col + count_border_pixels, num_cols - 1)

    return {
        "left": int(min_col),
        "top": int(min_row),
        "right": int(max_col),
        "bottom": int(max_row),
        "original_image_height": num_rows,
        "original_image_width": num_cols
    }

def remove_noise(image_original: np.ndarray):
    if image_original.ndim == 3:
        image = image_original[:, :, 0]
    else:
        image = image_original

    # Otsu's Thresholding
    # This automatically finds the best threshold value to separate background from foreground
    _, image_otsu = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Removes small "salt and pepper" noise from the thresholded mask
    kernel = np.ones((3,3), np.uint8)
    image_otsu = cv2.morphologyEx(image_otsu, cv2.MORPH_OPEN, kernel)
    image_clean = cv2.medianBlur(image_otsu, 5)

    return image_clean
