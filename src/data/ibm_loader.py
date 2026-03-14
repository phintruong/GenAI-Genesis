"""Load IBM AML transactions from local Kaggle cache."""

from pathlib import Path

import pandas as pd

from src.config import DATASET_DIR, DEFAULT_DATASET_FILE


def get_dataset_path(file_name=None):
    """Resolve transaction CSV path from configured local dataset directory."""
    name = file_name or DEFAULT_DATASET_FILE
    csv_path = DATASET_DIR / name
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. "
            "Make sure the dataset exists under kagglehub_cache."
        )
    return csv_path


def load_transactions(file_name=None):
    """Read selected transaction CSV into a pandas DataFrame."""
    path = get_dataset_path(file_name=file_name)
    return pd.read_csv(path)
