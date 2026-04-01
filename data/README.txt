Fruits-360 data for this project (Kaggle only)
==============================================

Source (only option used in this repo):
  https://www.kaggle.com/datasets/moltean/fruits

The code expects a folder that contains exactly:

  Training/
  Test/

Each subfolder is one class. Kaggle folder names often differ from classes.txt; the mapping
from your assigned labels to real folder names is in config.py -> CLASS_FOLDER_BY_SHORT_NAME.
Edit that dict if your unzip uses different names.

Default path (see config.py):
  Fruit360/data/Fruit-Images-Dataset/

You will NOT see any images here until you download from Kaggle. The folder is empty on purpose.


Quick setup
-----------

1) Kaggle API (recommended for scripts)

   - Install:  py -3 -m pip install kaggle
   - From Kaggle: Account -> Settings -> API -> Create New Token -> save kaggle.json
   - On Windows, place it at:  %USERPROFILE%\.kaggle\kaggle.json

2) Download and layout

   Option A — helper script (from project root):

     py -3 fetch_dataset.py

   The script finds kaggle.exe next to your Python (pip puts it in Scripts/, often not on PATH).
   If download still fails, run the CLI once to confirm auth:

     py -3 -c "import pathlib,sys; print(pathlib.Path(sys.executable).parent/'Scripts'/'kaggle.exe')"

   Paste that path in quotes, then:  ...\kaggle.exe datasets files -d moltean/fruits

   Option B — manual shell:

     kaggle datasets download -d moltean/fruits -p data\_kaggle_tmp --unzip

   After unzip, locate the folder that directly contains BOTH "Training" and "Test"
   (names inside the zip can vary). Point the code at it in one of two ways:

   - Rename/move that folder to:  data\Fruit-Images-Dataset

   - Or set an environment variable to that folder (if you keep the data elsewhere):

     PowerShell:
       $env:FRUIT360_DATA_ROOT = "C:\full\path\to\folder\with\Training\and\Test"


Check
-----

You should have paths like:

  ...\Training\Apple Crimson Snow\*.jpg
  ...\Test\Apple Crimson Snow\*.jpg

Run:  py -3 main.py --section bovw

If paths are wrong, you will get FileNotFoundError from dataset_utils.py.
