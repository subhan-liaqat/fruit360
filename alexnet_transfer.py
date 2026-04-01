"""Sections 5–6: AlexNet transfer learning (PyTorch)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.svm import SVC
from torchvision import models, transforms
from torchvision.models import AlexNet_Weights

from config import RANDOM_STATE, ensure_output_dir
from dataset_utils import assert_data_ready, load_split_paths


BATCH_SIZE = 32
FT_EPOCHS = 20
IMG_NET_SIZE = 224


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _imagenet_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_NET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _paths_to_tensors(paths: list[Path], tfm: transforms.Compose) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        xs.append(tfm(img))
    return torch.stack(xs, dim=0)


def _error_rate(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred != y).float().mean().item())


def section5a_pretrained_imagenet_heads(test_paths: list[Path], y_test: np.ndarray) -> dict:
    """5(a): Pretrained AlexNet with original ImageNet classifier (1000-way)."""
    device = _device()
    weights = AlexNet_Weights.IMAGENET1K_V1
    model = models.alexnet(weights=weights).to(device)
    model.eval()
    tfm = _imagenet_transform()
    y = torch.from_numpy(y_test).long().to(device)
    correct = 0
    n = 0
    with torch.no_grad():
        for i in range(0, len(test_paths), BATCH_SIZE):
            batch_p = test_paths[i : i + BATCH_SIZE]
            xb = _paths_to_tensors(batch_p, tfm).to(device)
            yb = y[i : i + BATCH_SIZE]
            logits = model(xb)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            n += len(batch_p)
    acc = correct / max(n, 1)
    err = 1.0 - acc
    out = {
        "note": "1000-way ImageNet logits vs 10 fruit labels — accuracy is not semantically meaningful.",
        "misleading_accuracy_vs_fruit_labels": acc,
        "error_treating_imagenet_argmax_as_fruit_class": err,
    }
    print("5(a): pretrained ImageNet classifier vs fruit labels (see note in outputs/alexnet_5a.json)")
    return out


def section5bc_frozen_head(
    train_paths: list[Path],
    y_train: np.ndarray,
    test_paths: list[Path],
    y_test: np.ndarray,
) -> dict:
    """5(b)(c): Freeze backbone, train new 10-class head for 20 epochs; plot train vs test error."""
    device = _device()
    weights = AlexNet_Weights.IMAGENET1K_V1
    model = models.alexnet(weights=weights).to(device)
    model.classifier[6] = nn.Linear(4096, 10)
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier[:6].parameters():
        p.requires_grad = False
    trainable = list(model.classifier[6].parameters())
    opt = torch.optim.Adam(trainable, lr=1e-3)
    crit = nn.CrossEntropyLoss()
    tfm = _imagenet_transform()

    X_train = _paths_to_tensors(train_paths, tfm).to(device)
    y_tr = torch.from_numpy(y_train).long().to(device)
    X_test = _paths_to_tensors(test_paths, tfm).to(device)
    y_te = torch.from_numpy(y_test).long().to(device)

    train_errs: list[float] = []
    test_errs: list[float] = []

    for _ in range(FT_EPOCHS):
        model.train()
        perm = torch.randperm(len(X_train), device=device)
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            opt.zero_grad()
            logits = model(X_train[idx])
            loss = crit(logits, y_tr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr_e = _error_rate(model(X_train), y_tr)
            te_e = _error_rate(model(X_test), y_te)
        train_errs.append(tr_e)
        test_errs.append(te_e)

    out_dir = ensure_output_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, FT_EPOCHS + 1), train_errs, "-o", label="Training error")
    ax.plot(range(1, FT_EPOCHS + 1), test_errs, "-s", label="Test error (used as validation)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.set_title("AlexNet frozen features + trained 10-class head")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "alexnet_frozen_head_curves.png", dpi=150)
    plt.close(fig)

    best_ep = int(np.argmin(test_errs)) + 1
    best_test_err = float(min(test_errs))
    summary = {
        "train_errors": train_errs,
        "test_errors": test_errs,
        "best_epoch_min_test_error": best_ep,
        "best_test_error": best_test_err,
    }
    with open(out_dir / "alexnet_5bc.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"5(b–c): best test error (frozen head) = {best_test_err:.4f} at epoch {best_ep}")
    return summary


def _extract_conv5_flat(paths: list[Path], model: nn.Module, device: torch.device) -> np.ndarray:
    """Last conv layer = features module index 10 (Conv2d) per torchvision AlexNet."""
    tfm = _imagenet_transform()
    feats: list[np.ndarray] = []
    hook_out: list[torch.Tensor] = []

    def hook(_m: nn.Module, _inp: torch.Tensor, out: torch.Tensor) -> None:
        hook_out.append(out.detach())

    h = model.features[10].register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[i : i + BATCH_SIZE]
            x = _paths_to_tensors(batch_paths, tfm).to(device)
            hook_out.clear()
            _ = model.features(x)
            t = hook_out[0]
            flat = t.flatten(1).cpu().numpy()
            feats.append(flat)
    h.remove()
    return np.vstack(feats)


def section5de_svm_on_conv5(
    train_paths: list[Path],
    y_train: np.ndarray,
    test_paths: list[Path],
    y_test: np.ndarray,
    kernel: str,
    C: float,
) -> dict:
    device = _device()
    weights = AlexNet_Weights.IMAGENET1K_V1
    model = models.alexnet(weights=weights).to(device)
    X_tr = _extract_conv5_flat(train_paths, model, device)
    X_te = _extract_conv5_flat(test_paths, model, device)
    kw: dict = {"kernel": kernel, "C": C, "random_state": RANDOM_STATE}
    if kernel in ("rbf", "poly", "sigmoid"):
        kw["gamma"] = "scale"
    clf = SVC(**kw)
    clf.fit(X_tr, y_train)
    tr_err = 1.0 - clf.score(X_tr, y_train)
    te_err = 1.0 - clf.score(X_te, y_test)
    out_dir = ensure_output_dir()
    summary = {
        "kernel": kernel,
        "C": C,
        "train_error": float(tr_err),
        "test_error": float(te_err),
        "feature_dim": X_tr.shape[1],
    }
    with open(out_dir / "alexnet_5de_svm_features.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"5(d–e): SVM on conv5 features — train err={tr_err:.4f}, test err={te_err:.4f}")
    return summary


def section6_finetune(
    train_paths: list[Path],
    y_train: np.ndarray,
    test_paths: list[Path],
    y_test: np.ndarray,
) -> dict:
    """6: Fine-tune full AlexNet for 20 epochs."""
    device = _device()
    weights = AlexNet_Weights.IMAGENET1K_V1
    model = models.alexnet(weights=weights).to(device)
    model.classifier[6] = nn.Linear(4096, 10)
    opt = torch.optim.Adam(
        [
            {"params": model.features.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ]
    )
    crit = nn.CrossEntropyLoss()
    tfm = _imagenet_transform()

    X_train = _paths_to_tensors(train_paths, tfm).to(device)
    y_tr = torch.from_numpy(y_train).long().to(device)
    X_test = _paths_to_tensors(test_paths, tfm).to(device)
    y_te = torch.from_numpy(y_test).long().to(device)

    train_errs: list[float] = []
    test_errs: list[float] = []

    for _ in range(FT_EPOCHS):
        model.train()
        perm = torch.randperm(len(X_train), device=device)
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            opt.zero_grad()
            logits = model(X_train[idx])
            loss = crit(logits, y_tr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr_e = _error_rate(model(X_train), y_tr)
            te_e = _error_rate(model(X_test), y_te)
        train_errs.append(tr_e)
        test_errs.append(te_e)

    out_dir = ensure_output_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, FT_EPOCHS + 1), train_errs, "-o", label="Training error")
    ax.plot(range(1, FT_EPOCHS + 1), test_errs, "-s", label="Test error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.set_title("Fine-tuned AlexNet (10-class)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "alexnet_finetune_curves.png", dpi=150)
    plt.close(fig)

    best_ep = int(np.argmin(test_errs)) + 1
    best_test_err = float(min(test_errs))
    summary = {
        "train_errors": train_errs,
        "test_errors": test_errs,
        "best_epoch_min_test_error": best_ep,
        "best_test_error": best_test_err,
    }
    with open(out_dir / "alexnet_finetune.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"6: fine-tune best test error = {best_test_err:.4f} at epoch {best_ep}")
    return summary


def run_alexnet_sections(svm_best: dict) -> dict:
    assert_data_ready()
    out_dir = ensure_output_dir()
    train_paths, y_train, test_paths, y_test, _ = load_split_paths()

    s5a = section5a_pretrained_imagenet_heads(test_paths, y_test)
    with open(out_dir / "alexnet_5a.json", "w", encoding="utf-8") as f:
        json.dump(s5a, f, indent=2)

    s5bc = section5bc_frozen_head(train_paths, y_train, test_paths, y_test)

    kernel = svm_best["kernel"]
    C = float(svm_best["C"])
    s5de = section5de_svm_on_conv5(train_paths, y_train, test_paths, y_test, kernel=kernel, C=C)

    s6 = section6_finetune(train_paths, y_train, test_paths, y_test)

    return {"5a": s5a, "5bc": s5bc, "5de": s5de, "6": s6}
