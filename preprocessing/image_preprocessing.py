"""Preprocessing utilities for retinal fundus images used by DRISHTI."""

from __future__ import annotations

import cv2
import numpy as np


TARGET_SIZE = (224, 224)


def crop_retina_region(image: np.ndarray) -> np.ndarray:
    """Crop the circular retina region when a clear fundus mask can be estimated.

    The method thresholds the grayscale image, finds the largest contour,
    and crops around it. If no contour is detected, the original image is returned.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cropped = image[y : y + h, x : x + w]
    return cropped if cropped.size else image


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement on the luminance channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_fundus_image(image: np.ndarray, target_size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Run the full preprocessing pipeline.

    Steps:
    1. Crop the retina region where possible.
    2. Apply Gaussian blur to suppress noise.
    3. Improve local contrast using CLAHE.
    4. Resize to the model input shape.
    5. Normalize pixels to the [0, 1] range.
    """
    cropped = crop_retina_region(image)
    blurred = cv2.GaussianBlur(cropped, (5, 5), 0)
    enhanced = apply_clahe(blurred)
    resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def load_and_preprocess_image(image_path: str, target_size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Load an image from disk and preprocess it for training or inference."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return preprocess_fundus_image(image, target_size)
