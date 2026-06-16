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

def convert_background_to_white(
    image_original: np.ndarray,
    dataset: str,
    closing_kernel_size: int = 5,
) -> np.ndarray:
    if image_original.ndim == 3:
        gray = image_original[:, :, 0].copy()
    else:
        gray = image_original.copy()

    if dataset.lower() == "hist_diy":
        # Liver dataset - where the background is fully black
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    elif dataset.lower() == "resc":
        # Otsu's Thresholding to separate the main object from the black background
        # RESC dataset
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)        

    # Keep only the largest external contour (the main object)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_original.copy()

    if dataset.lower() == "hist_diy":
        # There are samples where we have separate components, so we take all th identified foreground contours
        filled_mask = np.zeros_like(mask)
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    elif dataset.lower() == "resc":
        largest_contour = max(contours, key=cv2.contourArea)
        # Fill the contour so dark anomalies inside the main object stay part of the foreground
        filled_mask = np.zeros_like(mask)
        cv2.drawContours(filled_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    image_white_bg = image_original.copy()

    image_white_bg[filled_mask == 0] = 255
    
    return image_white_bg


def get_binary_threshold_map(image_original: np.ndarray, denoise_before=False):
    if image_original.ndim == 3:
        gray = image_original[:, :, 0].copy()
    else:
        gray = image_original.copy()

    # Otsu's Thresholding
    # This automatically finds the best threshold value to separate background from foreground
    if denoise_before:
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        # blurred = cv2.medianBlur(gray, 5)
    else:
        blurred = gray.copy()

    _, image_otsu = cv2.threshold(
        blurred, 
        0,
        255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Removes small "salt and pepper" noise from the thresholded mask
    kernel = np.ones((3,3), np.uint8)
    image_otsu = cv2.morphologyEx(image_otsu, cv2.MORPH_OPEN, kernel)
    binary_threshold_map = cv2.medianBlur(image_otsu, 5)

    return binary_threshold_map
