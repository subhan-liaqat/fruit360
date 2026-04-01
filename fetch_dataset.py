"""
Download Fruits-360 from Kaggle (moltean/fruits) and place Training/ + Test/ under:

  data/Fruit-Images-Dataset/

Requires:
  - pip install kaggle
  - Kaggle API token: ~/.kaggle/kaggle.json (Windows: %USERPROFILE%\\.kaggle\\kaggle.json)

  py -3 fetch_dataset.py

Manual download from the dataset page is fine; see data/README.txt.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_DATASET = "moltean/fruits"


def _kaggle_executable() -> str:
    """
    Resolve the Kaggle CLI. Pip installs kaggle.exe under Python's Scripts/
    folder, which is often not on PATH — then `kaggle` fails in subprocess/Git Bash.
    """
    bindir = Path(sys.executable).resolve().parent
    if sys.platform == "win32":
        for name in (bindir / "Scripts" / "kaggle.exe", bindir / "kaggle.exe"):
            if name.is_file():
                return str(name)
    else:
        candidate = bindir / "kaggle"
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("kaggle")
    if found:
        return found
    raise FileNotFoundError(
        "kaggle CLI not found next to Python and not on PATH. "
        "Install: py -3 -m pip install kaggle"
    )


def _find_training_test_root(search_under: Path) -> Path | None:
    """Return the parent folder that contains both Training/ and Test/ directories."""
    for training in search_under.rglob("Training"):
        if not training.is_dir():
            continue
        parent = training.parent
        test_dir = parent / "Test"
        if test_dir.is_dir():
            return parent
    return None


def main() -> None:
    root = Path(__file__).resolve().parent
    dest = root / "data" / "Fruit-Images-Dataset"
    if (dest / "Training").is_dir() and (dest / "Test").is_dir():
        print("Dataset already present:", dest)
        return

    staging = root / "data" / "_kaggle_staging"
    staging.mkdir(parents=True, exist_ok=True)

    try:
        kaggle_exe = _kaggle_executable()
    except FileNotFoundError as e:
        print("ERROR:", e, file=sys.stderr)
        print("Then add your API token: %USERPROFILE%\\.kaggle\\kaggle.json — see data/README.txt", file=sys.stderr)
        sys.exit(1)

    print("Downloading from Kaggle:", KAGGLE_DATASET)
    print("Using Kaggle CLI:", kaggle_exe)
    try:
        subprocess.run(
            [
                kaggle_exe,
                "datasets",
                "download",
                "-d",
                KAGGLE_DATASET,
                "-p",
                str(staging),
                "--unzip",
            ],
            check=True,
        )
    except FileNotFoundError:
        print("ERROR: could not run Kaggle CLI.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("ERROR: kaggle download failed:", e, file=sys.stderr)
        sys.exit(1)

    found = _find_training_test_root(staging)
    if found is None:
        print(
            "ERROR: Could not find a folder containing both Training/ and Test/ under",
            staging,
            file=sys.stderr,
        )
        print("Unzip manually, then set FRUIT360_DATA_ROOT or move that folder to:", dest, file=sys.stderr)
        sys.exit(1)

    print("Found dataset root:", found)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(found), str(dest))
    print("Ready:", dest)

    # remove empty staging dirs (ignore errors)
    try:
        shutil.rmtree(staging)
    except OSError:
        pass


if __name__ == "__main__":
    main()
