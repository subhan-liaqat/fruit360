"""Section 4: stratified 10-fold CNN learning curves and final test evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras import backend as K

from cnn_model import build_cnn
from config import RANDOM_STATE, ensure_output_dir
from dataset_utils import assert_data_ready, load_rgb_array, load_split_paths


N_SPLITS = 10
EPOCHS = 20
BATCH_SIZE = 32
IMG_SIZE = (100, 100)


def run_cnn_section(svm_best_test_error: float | None = None) -> dict:
    assert_data_ready()
    out = ensure_output_dir()

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    train_paths, y_train, test_paths, y_test, _ = load_split_paths()
    X_train = load_rgb_array(train_paths, size=IMG_SIZE)
    X_test = load_rgb_array(test_paths, size=IMG_SIZE)

    y_train_cat = keras.utils.to_categorical(y_train, num_classes=10)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes=10)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    train_err_folds = np.zeros((N_SPLITS, EPOCHS))
    val_err_folds = np.zeros((N_SPLITS, EPOCHS))

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        K.clear_session()
        model = build_cnn(input_shape=(*IMG_SIZE, 3), num_classes=10)
        hist = model.fit(
            X_train[tr_idx],
            y_train_cat[tr_idx],
            validation_data=(X_train[va_idx], y_train_cat[va_idx]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        tr_acc = np.asarray(hist.history["accuracy"], dtype=np.float64)
        va_acc = np.asarray(hist.history["val_accuracy"], dtype=np.float64)
        train_err_folds[fold_idx] = 1.0 - tr_acc
        val_err_folds[fold_idx] = 1.0 - va_acc

    mean_train = train_err_folds.mean(axis=0)
    std_train = train_err_folds.std(axis=0)
    mean_val = val_err_folds.mean(axis=0)
    std_val = val_err_folds.std(axis=0)

    epochs_x = np.arange(1, EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(epochs_x, mean_train, yerr=std_train, fmt="-o", capsize=3, label="Training error")
    ax.errorbar(epochs_x, mean_val, yerr=std_val, fmt="-s", capsize=3, label="Validation error (10-fold)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.set_title("CNN learning curves (mean ± std over stratified 10-fold)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cnn_learning_curves.png", dpi=150)
    plt.close(fig)

    best_epoch_idx = int(np.argmin(mean_val))
    best_epochs = best_epoch_idx + 1

    K.clear_session()
    final = build_cnn(input_shape=(*IMG_SIZE, 3), num_classes=10)
    final.fit(
        X_train,
        y_train_cat,
        epochs=best_epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    _, test_acc = final.evaluate(X_test, y_test_cat, verbose=0)
    test_error = float(1.0 - test_acc)

    result = {
        "best_epoch": best_epochs,
        "test_error": test_error,
        "mean_val_per_epoch": mean_val.tolist(),
        "svm_best_test_error_for_report": svm_best_test_error,
    }
    with open(out / "cnn_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"CNN: best epoch (min mean val error) = {best_epochs}, test error = {test_error:.4f}")
    if svm_best_test_error is not None:
        print(f"Best BoVW+SVM test error (from Section 3): {svm_best_test_error:.4f}")

    return result
