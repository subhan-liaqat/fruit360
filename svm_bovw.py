"""Section 3: SVM on 100-D BoVW with stratified 10-fold CV and test evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from bovw import histograms_for_paths
from config import RANDOM_STATE, ensure_output_dir
from dataset_utils import assert_data_ready, load_split_paths


C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
KERNELS = ["linear", "rbf", "poly", "sigmoid"]
N_SPLITS = 10


def _errors_clf(clf: SVC, X: np.ndarray, y: np.ndarray) -> float:
    return float(1.0 - clf.score(X, y))


def _svc(kernel: str, C: float) -> SVC:
    kw: dict = {"kernel": kernel, "C": C, "random_state": RANDOM_STATE}
    if kernel in ("rbf", "poly", "sigmoid"):
        kw["gamma"] = "scale"
    return SVC(**kw)


def run_svm_bovw(bovw_pack: dict) -> dict:
    assert_data_ready()
    out = ensure_output_dir()

    X_train = np.asarray(bovw_pack["histograms_train"], dtype=np.float64)
    y_train = np.asarray(bovw_pack["y_train"])
    kmeans = bovw_pack["kmeans"]
    sift = bovw_pack["sift"]

    train_paths, _, test_paths, y_test, _ = load_split_paths()
    X_test = histograms_for_paths(test_paths, kmeans, sift)
    y_test = np.asarray(y_test)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results_by_kernel: dict = {}

    for kernel in KERNELS:
        mean_train = []
        std_train = []
        mean_val = []
        std_val = []
        test_errs = []

        for C in C_VALUES:
            fold_train_err = []
            fold_val_err = []
            for tr_idx, va_idx in skf.split(X_train, y_train):
                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                clf = _svc(kernel, C)
                clf.fit(X_tr, y_tr)
                fold_train_err.append(_errors_clf(clf, X_tr, y_tr))
                fold_val_err.append(_errors_clf(clf, X_va, y_va))

            mean_train.append(float(np.mean(fold_train_err)))
            std_train.append(float(np.std(fold_train_err)))
            mean_val.append(float(np.mean(fold_val_err)))
            std_val.append(float(np.std(fold_val_err)))

            clf_full = _svc(kernel, C)
            clf_full.fit(X_train, y_train)
            test_errs.append(_errors_clf(clf_full, X_test, y_test))

        results_by_kernel[kernel] = {
            "C_VALUES": C_VALUES,
            "mean_train": mean_train,
            "std_train": std_train,
            "mean_val": mean_val,
            "std_val": std_val,
            "test_error": test_errs,
        }

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(C_VALUES))
        ax.errorbar(
            x,
            mean_train,
            yerr=std_train,
            fmt="-o",
            capsize=4,
            label="Training error",
        )
        ax.errorbar(
            x,
            mean_val,
            yerr=std_val,
            fmt="-s",
            capsize=4,
            label="Validation error (10-fold)",
        )
        ax.plot(x, test_errs, "-^", label="Test error (full train)", color="tab:red")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in C_VALUES])
        ax.set_xlabel("C")
        ax.set_ylabel("Error")
        ax.set_title(f"SVM BoVW — kernel={kernel}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"svm_bovw_{kernel}.png", dpi=150)
        plt.close(fig)

    best_by_kernel: dict[str, dict] = {}
    for kernel in KERNELS:
        te = results_by_kernel[kernel]["test_error"]
        best_i = int(np.argmin(te))
        best_C = C_VALUES[best_i]
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        fold_train_err = []
        fold_val_err = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            clf = _svc(kernel, best_C)
            clf.fit(X_tr, y_tr)
            fold_train_err.append(_errors_clf(clf, X_tr, y_tr))
            fold_val_err.append(_errors_clf(clf, X_va, y_va))

        clf_full = _svc(kernel, best_C)
        clf_full.fit(X_train, y_train)
        test_te = _errors_clf(clf_full, X_test, y_test)

        best_by_kernel[kernel] = {
            "best_C": best_C,
            "mean_train": float(np.mean(fold_train_err)),
            "std_train": float(np.std(fold_train_err)),
            "mean_val": float(np.mean(fold_val_err)),
            "std_val": float(np.std(fold_val_err)),
            "test_error": test_te,
        }

    kernels_order = KERNELS
    xk = np.arange(len(kernels_order))
    width = 0.25
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    mt = [best_by_kernel[k]["mean_train"] for k in kernels_order]
    st = [best_by_kernel[k]["std_train"] for k in kernels_order]
    mv = [best_by_kernel[k]["mean_val"] for k in kernels_order]
    sv = [best_by_kernel[k]["std_val"] for k in kernels_order]
    tst = [best_by_kernel[k]["test_error"] for k in kernels_order]

    ax2.bar(xk - width, mt, width, yerr=st, capsize=3, label="Train (CV mean)", color="tab:blue")
    ax2.bar(xk, mv, width, yerr=sv, capsize=3, label="Validation (CV mean)", color="tab:orange")
    ax2.bar(xk + width, tst, width, label="Test (refit all train)", color="tab:red")
    ax2.set_xticks(xk)
    ax2.set_xticklabels(kernels_order)
    ax2.set_ylabel("Error")
    ax2.set_title("Best SVM per kernel (lowest test error over C)")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out / "svm_bovw_best_kernel_comparison.png", dpi=150)
    plt.close(fig2)

    summary = {
        "results_by_kernel": results_by_kernel,
        "best_by_kernel": best_by_kernel,
    }
    with open(out / "svm_bovw_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    best_kernel = min(best_by_kernel, key=lambda k: best_by_kernel[k]["test_error"])
    print("Best kernel (lowest test error):", best_kernel, best_by_kernel[best_kernel])
    return {**summary, "best_kernel_name": best_kernel, "best_kernel_info": best_by_kernel[best_kernel]}
