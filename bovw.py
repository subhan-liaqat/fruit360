"""Section 2: SIFT, bag-of-visual-words (K=100), PCA visualization."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from config import RANDOM_STATE, ensure_output_dir
from dataset_utils import assert_data_ready, load_split_paths


N_VISUAL_WORDS = 100
SIFT_DESCRIPTOR_DIM = 128


def _read_gray(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def extract_sift_for_image(sift: cv2.SIFT, gray: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    kps, des = sift.detectAndCompute(gray, None)
    if des is None:
        return [], np.empty((0, SIFT_DESCRIPTOR_DIM), dtype=np.float32)
    return list(kps), des.astype(np.float32)


def run_bovw_and_pca() -> dict:
    assert_data_ready()
    out = ensure_output_dir()
    train_paths, y_train, _, _, short_names = load_split_paths()

    sift = cv2.SIFT_create()
    all_descriptors: list[np.ndarray] = []
    per_image_descriptors: list[np.ndarray] = []
    sample_viz_path = train_paths[0]
    sample_kps: list = []
    for p in train_paths:
        g = _read_gray(p)
        kps, _ = extract_sift_for_image(sift, g)
        if kps:
            sample_viz_path = p
            sample_kps = kps
            sample_gray = g
            break
    else:
        sample_gray = _read_gray(train_paths[0])

    # Keypoint visualization (one training image)
    vis = cv2.drawKeypoints(
        sample_gray,
        sample_kps,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    fig0, ax0 = plt.subplots(figsize=(8, 8))
    ax0.imshow(vis, cmap="gray")
    ax0.set_title(f"SIFT keypoints: {sample_viz_path.name}")
    ax0.axis("off")
    fig0.tight_layout()
    fig0.savefig(out / "sift_keypoints.png", dpi=150)
    plt.close(fig0)

    for p in train_paths:
        gray = _read_gray(p)
        _, des = extract_sift_for_image(sift, gray)
        per_image_descriptors.append(des)
        if des.shape[0] > 0:
            all_descriptors.append(des)

    if not all_descriptors:
        raise RuntimeError("No SIFT descriptors extracted; check images and OpenCV SIFT build.")

    kp_matrix = np.vstack(all_descriptors)
    print(f"Total keypoint descriptors for clustering: {kp_matrix.shape[0]}")

    kmeans = MiniBatchKMeans(
        n_clusters=N_VISUAL_WORDS,
        random_state=RANDOM_STATE,
        batch_size=4096,
        n_init=3,
        max_iter=100,
    )
    kmeans.fit(kp_matrix)

    n_train = len(train_paths)
    histograms = np.zeros((n_train, N_VISUAL_WORDS), dtype=np.float64)
    for i, des in enumerate(per_image_descriptors):
        if des.shape[0] == 0:
            continue
        labels = kmeans.predict(des)
        histograms[i] = np.bincount(labels, minlength=N_VISUAL_WORDS)

    row_sums = histograms.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    histograms = histograms / row_sums

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    D_2d = pca.fit_transform(histograms)

    fig, ax = plt.subplots(figsize=(10, 8))
    markers = ["o", "s", "^", "v", "D", "P", "*", "X", "h", "8"]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for c in range(len(short_names)):
        mask = y_train == c
        ax.scatter(
            D_2d[mask, 0],
            D_2d[mask, 1],
            c=[colors[c]],
            marker=markers[c % len(markers)],
            label=short_names[c],
            alpha=0.7,
            s=25,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA (2D) of 100-D bag-of-visual-words (training set)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pca_bovw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "histograms_train": histograms,
        "y_train": y_train,
        "short_names": short_names,
        "kmeans": kmeans,
        "sift": sift,
    }


def histograms_for_paths(paths: list[Path], kmeans: MiniBatchKMeans, sift: cv2.SIFT) -> np.ndarray:
    """100-D BoVW vectors for arbitrary image paths (same pipeline as training)."""
    n = len(paths)
    h = np.zeros((n, N_VISUAL_WORDS), dtype=np.float64)
    for i, p in enumerate(paths):
        gray = _read_gray(p)
        _, des = extract_sift_for_image(sift, gray)
        if des.shape[0] == 0:
            continue
        labels = kmeans.predict(des)
        h[i] = np.bincount(labels, minlength=N_VISUAL_WORDS)
    row_sums = h.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return h / row_sums
