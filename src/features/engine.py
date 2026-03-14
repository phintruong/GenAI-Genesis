"""Feature preparation for IBM AML transaction data."""

import pandas as pd

from src.config import IBM_REQUIRED_COLUMNS, MODEL_FEATURE_COLUMNS


def validate_ibm_schema(df):
    """Validate incoming DataFrame contains required IBM AML columns."""
    missing = [c for c in IBM_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_model_matrix(df):
    """Build X/y model inputs from raw IBM AML transaction rows."""
    validate_ibm_schema(df)

    X = df[
        [
            "Timestamp",
            "From Bank",
            "Account",
            "To Bank",
            "Account.1",
            "Amount Received",
            "Receiving Currency",
            "Amount Paid",
            "Payment Currency",
            "Payment Format",
        ]
    ].copy()

    y = pd.to_numeric(df["Is Laundering"], errors="coerce").fillna(0).astype(int)

    ts = pd.to_datetime(X["Timestamp"], errors="coerce")
    X["Hour"] = ts.dt.hour.fillna(-1).astype("int16")
    X["DayOfWeek"] = ts.dt.dayofweek.fillna(-1).astype("int16")
    X["Day"] = ts.dt.day.fillna(-1).astype("int16")
    X["Month"] = ts.dt.month.fillna(-1).astype("int16")
    X = X.drop(columns=["Timestamp"])

    numeric_cols = ["From Bank", "To Bank", "Amount Received", "Amount Paid"]
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    categorical_cols = [
        "Account",
        "Account.1",
        "Receiving Currency",
        "Payment Currency",
        "Payment Format",
    ]
    for col in categorical_cols:
        X[col] = pd.factorize(X[col], sort=True)[0].astype("int32")

    X = X[MODEL_FEATURE_COLUMNS].copy()
    return X, y
