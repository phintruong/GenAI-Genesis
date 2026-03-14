"""Model training and evaluation for IBM AML Random Forest baseline."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import MAX_ROWS, RANDOM_STATE, RF_N_ESTIMATORS, TEST_SIZE


def stratified_downsample(X, y, max_rows=MAX_ROWS, random_state=RANDOM_STATE):
    """Downsample while preserving class ratio for faster training."""
    if len(X) <= max_rows:
        return X, y

    frac = max_rows / len(X)
    sampled_idx = y.groupby(y, group_keys=False).apply(
        lambda s: s.sample(frac=frac, random_state=random_state)
    ).index
    return X.loc[sampled_idx], y.loc[sampled_idx]


def train_random_forest(X, y):
    """Train RandomForest and return model plus split datasets."""
    X_model, y_model = stratified_downsample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_model,
        y_model,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_model,
    )

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Compute core classification metrics and return structured results."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "classification_report": classification_report(y_test, y_pred, digits=4, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "test_size": len(X_test),
    }
