"""Image quality checks for uploaded or captured retinal fundus photographs."""

from __future__ import annotations

import cv2
import numpy as np


BLUR_THRESHOLD = 80.0
BRIGHTNESS_LOW = 45.0
BRIGHTNESS_HIGH = 220.0
CENTER_ALIGNMENT_THRESHOLD = 0.35


def blur_score(image: np.ndarray) -> float:
    """Estimate focus quality using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(image: np.ndarray) -> float:
    """Estimate average brightness using the HSV value channel."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


def center_alignment_score(image: np.ndarray) -> float:
    """Approximate whether the brightest retinal structure is near the center.

    This is a lightweight proxy for checking whether the optic-disc region is
    roughly centered in the frame.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)

    height, width = gray.shape
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    peak = np.array(max_loc, dtype=np.float32)
    distance = np.linalg.norm(center - peak)
    diagonal = np.linalg.norm(np.array([width, height], dtype=np.float32))
    normalized_distance = distance / diagonal
    return max(0.0, 1.0 - normalized_distance * 2.0)


def evaluate_image_quality(image: np.ndarray) -> dict:
    """Return detailed image-quality metrics and an accept/reject decision."""
    blur = blur_score(image)
    brightness = brightness_score(image)
    alignment = center_alignment_score(image)

    accepted = (
        blur >= BLUR_THRESHOLD
        and BRIGHTNESS_LOW <= brightness <= BRIGHTNESS_HIGH
        and alignment >= CENTER_ALIGNMENT_THRESHOLD
    )

    return {
        "accepted": bool(accepted),
        "blur_score": blur,
        "brightness_score": brightness,
        "center_alignment_score": alignment,
        "reasons": {
            "is_blurry": blur < BLUR_THRESHOLD,
            "is_too_dark": brightness < BRIGHTNESS_LOW,
            "is_too_bright": brightness > BRIGHTNESS_HIGH,
            "is_off_center": alignment < CENTER_ALIGNMENT_THRESHOLD,
        },
    }
