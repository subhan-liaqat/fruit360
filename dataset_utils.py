"""Load image paths and integer labels for Training/Test folders."""
from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np

from config import CLASS_FOLDER_BY_SHORT_NAME, TEST_DIR, TRAINING_DIR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")
    paths: list[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS:
            paths.append(p)
    return paths


def load_split_paths(
    min_train_images: int = 400,
) -> tuple[list[Path], np.ndarray, list[Path], np.ndarray, list[str]]:
    """
    Returns training paths, y_train, test paths, y_test, short class names (order = label index).
    """
    short_names = list(CLASS_FOLDER_BY_SHORT_NAME.keys())
    name_to_idx = {n: i for i, n in enumerate(short_names)}

    train_paths: list[Path] = []
    train_labels: list[int] = []
    test_paths: list[Path] = []
    test_labels: list[int] = []

    for short, folder_name in CLASS_FOLDER_BY_SHORT_NAME.items():
        tr_dir = TRAINING_DIR / folder_name
        te_dir = TEST_DIR / folder_name
        tr_imgs = list_images(tr_dir)
        te_imgs = list_images(te_dir)
        if len(tr_imgs) < min_train_images:
            warnings.warn(
                f"{folder_name}: only {len(tr_imgs)} training images "
                f"(assignment suggests >{min_train_images}).",
                stacklevel=2,
            )
        idx = name_to_idx[short]
        train_paths.extend(tr_imgs)
        train_labels.extend([idx] * len(tr_imgs))
        test_paths.extend(te_imgs)
        test_labels.extend([idx] * len(te_imgs))

    return (
        train_paths,
        np.array(train_labels, dtype=np.int64),
        test_paths,
        np.array(test_labels, dtype=np.int64),
        short_names,
)


def assert_data_ready() -> None:
    if not TRAINING_DIR.is_dir() or not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected dataset at {TRAINING_DIR.parent}. "
            "Clone https://github.com/Horea94/Fruit-Images-Dataset into data/Fruit-Images-Dataset"
        )


def load_rgb_array(paths: list[Path], size: tuple[int, int] = (100, 100)) -> np.ndarray:
    """RGB images resized to `size`, float32 in [0, 1], shape (N, H, W, 3)."""
    h, w = size
    out = np.zeros((len(paths), h, w, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise ValueError(f"Could not read: {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != h or rgb.shape[1] != w:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        out[i] = rgb.astype(np.float32) / 255.0
    return out
