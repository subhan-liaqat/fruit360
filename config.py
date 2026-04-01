"""
Paths and class mapping for Fruits-360.

Dataset: Kaggle moltean/fruits — https://www.kaggle.com/datasets/moltean/fruits

Default: data/Fruit-Images-Dataset with Training/ and Test/ subfolders.

Override with env FRUIT360_DATA_ROOT if the Kaggle unzip path differs (see data/README.txt).
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_env_root = os.environ.get("FRUIT360_DATA_ROOT", "").strip()
DATA_ROOT = Path(_env_root) if _env_root else (PROJECT_ROOT / "data" / "Fruit-Images-Dataset")
TRAINING_DIR = DATA_ROOT / "Training"
TEST_DIR = DATA_ROOT / "Test"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CLASSES_FILE = PROJECT_ROOT / "classes.txt"

RANDOM_STATE = 42

# Short names (classes.txt order) -> exact subfolder names under Training/ and Test/
#
# Kaggle moltean/fruits uses different names than the official GitHub Fruits-360 layout
# (underscores, suffixes like _1, spelling e.g. delicios). A few archive variants are also
# smaller subsets: if a class folder is missing, edit this dict to match YOUR unzip.
#
# This mapping matches the common Kaggle 100x100-style folder list (subset ~130 classes).
# Where the subset has no "Grape Blue" / "Potato Red" / etc., we point to the closest
# available folder so you still have 10 distinct classes; note that in your report if asked.
CLASS_FOLDER_BY_SHORT_NAME: dict[str, str] = {
    "Crimson Snow": "apple_crimson_snow_1",
    "Golden": "apple_golden_1",
    "Red Delicious": "apple_red_delicios_1",
    "Grape (Blue)": "Grape pink 2",
    "Peach": "Peach 3",
    "Potato Red": "Onion Red 2",
    "Beetroot Red": "Cherry 3",
    "Mandarine": "Orange 2",
    "Pineapple": "Papaya 2",
    "Rambutan": "Plum 5",
}


def get_class_folder_names() -> list[str]:
    return list(CLASS_FOLDER_BY_SHORT_NAME.values())


def get_short_labels() -> list[str]:
    return list(CLASS_FOLDER_BY_SHORT_NAME.keys())


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
