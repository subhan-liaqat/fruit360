Fruits-360 data (Kaggle)
========================

Run the assignment from:  ML1_Fruit360_Assignment.ipynb  (project root).

Dataset page:
  https://www.kaggle.com/datasets/moltean/fruits

Expected layout under the project:

  data/Fruit-Images-Dataset/Training/<class folders>/
  data/Fruit-Images-Dataset/Test/<class folders>/

Download: Kaggle website (Download) or CLI, e.g.:

  kaggle datasets download -d moltean/fruits -p data\_kaggle_tmp --unzip

Then move/rename so Training/ and Test/ sit under data\Fruit-Images-Dataset\, or set:

  $env:FRUIT360_DATA_ROOT = "C:\path\to\folder\containing\Training\and\Test"

Folder names must match the dict CLASS_FOLDER_BY_SHORT_NAME in the first code cell
of the notebook (edit there if your zip uses different names).
