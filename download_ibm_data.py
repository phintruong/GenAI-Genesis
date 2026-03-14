"""
Download the IBM AML transaction dataset so run_pipeline.py can find it.

Run once:  python download_ibm_data.py

Requires:  pip install kagglehub
Kaggle:    https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
You may need to accept the dataset terms on Kaggle and have credentials set up.
"""

import os
import shutil
from pathlib import Path

# Set cache before importing kagglehub so it uses project folder
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE = PROJECT_ROOT / "kagglehub_cache"
DATA_DIR = PROJECT_ROOT / "Data"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ["KAGGLEHUB_CACHE"] = str(CACHE.resolve())

import kagglehub

DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
DEFAULT_FILE = "HI-Small_Trans.csv"


def main():
    print("Downloading IBM AML dataset (this may take a moment)...")
    path_str = kagglehub.dataset_download(DATASET)
    path = Path(path_str)
    print(f"Downloaded to: {path}")

    # Copy default CSV into Data/ so pipeline finds it without relying on cache layout
    src = path / DEFAULT_FILE
    if src.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dst = DATA_DIR / DEFAULT_FILE
        shutil.copy2(src, dst)
        print(f"Copied {DEFAULT_FILE} to {DATA_DIR}")
        print("You can now run:  python run_pipeline.py")
    else:
        available = list(path.iterdir())
        print(f"Expected {DEFAULT_FILE} not found. Available: {[p.name for p in available]}")
        print("Run:  python run_pipeline.py --csv", path / (available[0].name if available else ""))


if __name__ == "__main__":
    main()
