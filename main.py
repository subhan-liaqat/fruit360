"""
ML1 Fruits-360 assignment driver.

Requires: Fruits-360 from Kaggle (moltean/fruits) — see data/README.txt and fetch_dataset.py

Example:
  py main.py --all
  py main.py --section all
  py main.py --section bovw
  py main.py --section svm cnn alexnet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bovw import run_bovw_and_pca
from config import OUTPUT_DIR, ensure_output_dir
from svm_bovw import run_svm_bovw

# TensorFlow (cnn_train) and heavy torch imports are loaded only when needed so
# `main.py --section bovw svm alexnet` works on Python 3.14 without TensorFlow.


def _write_assignment_summary(
    svm_pack: dict,
    cnn_pack: dict | None,
    alex_pack: dict | None,
) -> None:
    out = ensure_output_dir()
    best_svm_test = float(svm_pack["best_kernel_info"]["test_error"])
    payload: dict = {
        "best_bovw_svm": {
            "kernel": svm_pack["best_kernel_name"],
            "C": svm_pack["best_kernel_info"]["best_C"],
            "test_error": best_svm_test,
        },
    }
    if cnn_pack:
        payload["cnn"] = {
            "best_epoch": cnn_pack["best_epoch"],
            "test_error": cnn_pack["test_error"],
        }
    if alex_pack:
        payload["alexnet_frozen_head_best_test"] = alex_pack["5bc"].get("best_test_error")
        payload["alexnet_conv5_svm_test"] = alex_pack["5de"].get("test_error")
        payload["alexnet_finetune_best_test"] = alex_pack["6"].get("best_test_error")
    with open(out / "assignment_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fruit-360 ML1 assignment pipeline")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all sections (bovw, svm, cnn, alexnet). Same idea as --section all.",
    )
    parser.add_argument(
        "--section",
        nargs="+",
        choices=["bovw", "svm", "cnn", "alexnet", "all"],
        default=["all"],
        help="Which parts to run (default: all)",
    )
    args = parser.parse_args()
    if args.all:
        sections = {"bovw", "svm", "cnn", "alexnet"}
    else:
        sections = set(args.section)
        if "all" in sections:
            sections = {"bovw", "svm", "cnn", "alexnet"}

    need_bovw = "bovw" in sections
    need_svm = "svm" in sections
    need_cnn = "cnn" in sections
    need_alex = "alexnet" in sections

    ensure_output_dir()

    bovw_dict = None
    if need_bovw or need_svm or need_cnn or need_alex:
        bovw_dict = run_bovw_and_pca()

    svm_pack = None
    if need_svm or need_cnn or need_alex:
        if bovw_dict is None:
            bovw_dict = run_bovw_and_pca()
        svm_pack = run_svm_bovw(bovw_dict)

    cnn_pack = None
    if need_cnn:
        try:
            from cnn_train import run_cnn_section
        except ModuleNotFoundError as e:
            if e.name in ("tensorflow", "keras"):
                raise SystemExit(
                    "CNN requires TensorFlow. Install Python 3.10–3.12, then:\n"
                    "  py -m pip install tensorflow\n"
                    "Or run without CNN:  py main.py --section bovw svm alexnet"
                ) from e
            raise
        best_svm = float(svm_pack["best_kernel_info"]["test_error"]) if svm_pack else None
        cnn_pack = run_cnn_section(svm_best_test_error=best_svm)

    alex_pack = None
    if need_alex:
        from alexnet_transfer import run_alexnet_sections

        if svm_pack is None:
            if bovw_dict is None:
                bovw_dict = run_bovw_and_pca()
            svm_pack = run_svm_bovw(bovw_dict)
        svm_best = {
            "kernel": svm_pack["best_kernel_name"],
            "C": svm_pack["best_kernel_info"]["best_C"],
        }
        alex_pack = run_alexnet_sections(svm_best)

    if svm_pack:
        _write_assignment_summary(svm_pack, cnn_pack, alex_pack)

    print(f"Done. Figures and JSON under {Path(OUTPUT_DIR).resolve()}")


if __name__ == "__main__":
    main()
