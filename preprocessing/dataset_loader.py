"""Dataset loading helpers for DRISHTI retinal fundus classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from preprocessing.image_preprocessing import load_and_preprocess_image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = ["glaucoma", "normal"]


@dataclass
class DatasetSplit:
    images: np.ndarray
    labels: np.ndarray
    paths: list[str]


def iter_image_paths(root_dir: str | Path) -> Iterable[tuple[str, int]]:
    """Yield image paths and numeric labels from a class-folder dataset."""
    root_path = Path(root_dir)
    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = root_path / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield str(image_path), label_index


def load_split(split_dir: str | Path) -> DatasetSplit:
    """Load and preprocess a dataset split from disk into memory."""
    images: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []

    for image_path, label in iter_image_paths(split_dir):
        images.append(load_and_preprocess_image(image_path))
        labels.append(label)
        paths.append(image_path)

    if not images:
        return DatasetSplit(
            images=np.empty((0, 224, 224, 3), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int64),
            paths=[],
        )

    return DatasetSplit(
        images=np.stack(images).astype(np.float32),
        labels=np.array(labels, dtype=np.int64),
        paths=paths,
    )
