"""
Paths and class mapping for Fruits-360 (Horea94/Fruit-Images-Dataset).

Clone the dataset into data/Fruit-Images-Dataset with Training/ and Test/ subfolders.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "Fruit-Images-Dataset"
TRAINING_DIR = DATA_ROOT / "Training"
TEST_DIR = DATA_ROOT / "Test"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CLASSES_FILE = PROJECT_ROOT / "classes.txt"

RANDOM_STATE = 42

# Short names (classes.txt order) -> exact folder names in Training/Test
CLASS_FOLDER_BY_SHORT_NAME: dict[str, str] = {
    "Crimson Snow": "Apple Crimson Snow",
    "Golden": "Apple Golden 1",
    "Red Delicious": "Apple Red Delicious",
    "Grape (Blue)": "Grape Blue",
    "Peach": "Peach",
    "Potato Red": "Potato Red",
    "Beetroot Red": "Beetroot",
    "Mandarine": "Mandarine",
    "Pineapple": "Pineapple",
    "Rambutan": "Rambutan",
}


def get_class_folder_names() -> list[str]:
    return list(CLASS_FOLDER_BY_SHORT_NAME.values())


def get_short_labels() -> list[str]:
    return list(CLASS_FOLDER_BY_SHORT_NAME.keys())


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
