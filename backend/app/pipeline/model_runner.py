"""Score transactions via local model (joblib / PyTorch GNN) or MODEL_URL microservice."""

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from app.config import MODEL_PATH, MODEL_URL

# Feature column order must match training (tabular path) and be available for GNN node features
FEATURE_COLUMNS = [
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Hour",
    "DayOfWeek",
    "Day",
    "Month",
]


def _is_torch_checkpoint(obj: Any) -> bool:
    """True if obj is a torch checkpoint dict (model_state_dict, config, input_dim, model_name, feature_set)."""
    if not isinstance(obj, dict):
        return False
    required = {"model_state_dict", "config", "input_dim", "model_name", "feature_set"}
    return required.issubset(obj.keys())


def _load_torch_checkpoint(path: str) -> Tuple[Any, Dict, int, str, str]:
    """Load torch.save checkpoint. Returns (model, config, input_dim, model_name, feature_set)."""
    import torch
    from app.pipeline.gnn_models import build_model

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return _build_gnn_from_checkpoint(ckpt)


def _build_gnn_from_checkpoint(ckpt: Dict) -> Tuple[Any, Dict, int, str, str]:
    """Build GNN model from already-loaded checkpoint dict."""
    from app.pipeline.gnn_models import build_model

    if not _is_torch_checkpoint(ckpt):
        raise ValueError("Not a GNN checkpoint: missing model_state_dict/config/input_dim/model_name/feature_set")

    config = ckpt["config"]
    input_dim = int(ckpt["input_dim"])
    model_name = ckpt["model_name"]
    feature_set = ckpt.get("feature_set", "")

    model = build_model(model_name, input_dim, config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, config, input_dim, model_name, feature_set


def _build_graph_from_df(df: pd.DataFrame, input_dim: int) -> Tuple[Any, Any, Dict[int, int], List[int]]:
    """
    Build graph from preprocessed DataFrame (Account, Account.1 are factorized integers).
    Nodes = unique (Account) U (Account.1). Edges = transactions. Node features = aggregate per node, padded to input_dim.
    Returns (x, edge_index, account_to_idx, account_order).
    """
    import torch

    src_col = "Account"
    dst_col = "Account.1" if "Account.1" in df.columns else "Account"
    all_ids = pd.Index(df[src_col].unique().tolist() + df[dst_col].unique().tolist()).unique()
    account_order = all_ids.tolist()
    account_to_idx = {aid: i for i, aid in enumerate(account_order)}
    n_nodes = len(account_order)

    use_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not use_cols:
        use_cols = [c for c in df.columns if df[c].dtype in (np.number, "int64", "float64")][:input_dim]
    feat_from = df.groupby(src_col)[use_cols].mean()
    feat_to = df.groupby(dst_col)[use_cols].mean() if dst_col in df.columns else feat_from
    agg_df = feat_from.reindex(account_order).fillna(feat_to.reindex(account_order)).fillna(0)
    feat = agg_df.values.astype(np.float32)
    if feat.shape[1] < input_dim:
        pad = np.zeros((n_nodes, input_dim - feat.shape[1]), dtype=np.float32)
        feat = np.hstack([feat, pad])
    elif feat.shape[1] > input_dim:
        feat = feat[:, :input_dim]
    x = torch.from_numpy(feat)

    src_idx = df[src_col].map(account_to_idx)
    dst_idx = df[dst_col].map(account_to_idx)
    valid = src_idx.notna() & dst_idx.notna()
    edge_index = np.stack([src_idx[valid].astype(int).values, dst_idx[valid].astype(int).values], axis=0)
    edge_index = torch.from_numpy(edge_index).long()

    return x, edge_index, account_to_idx, account_order


def _score_gnn(df: pd.DataFrame, model, input_dim: int) -> pd.DataFrame:
    """Score using GNN: build graph, run forward, map node scores back to transaction rows."""
    import torch

    x, edge_index, account_to_idx, _ = _build_graph_from_df(df, input_dim)

    with torch.no_grad():
        log_probs = model(x, edge_index)
        prob_pos = log_probs.exp()[:, 1].numpy()

    # Map: each transaction gets the risk of its "from" account (Account = factorized int)
    node_idx = df["Account"].map(account_to_idx)
    node_idx = node_idx.fillna(0).astype(int).clip(0, len(prob_pos) - 1)
    risk_scores = np.clip(prob_pos[node_idx.values], 0.0, 1.0).astype(np.float64)

    out = df.copy()
    out["risk_score"] = risk_scores
    out["top_features"] = [[]] * len(out)
    return out


def _get_model():
    """Load model from MODEL_PATH: either joblib (sklearn) or torch checkpoint (GNN)."""
    if not MODEL_PATH or not MODEL_PATH.strip():
        raise RuntimeError("MODEL_PATH is not set")
    try:
        import torch
        obj = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        if _is_torch_checkpoint(obj):
            model, config, input_dim, model_name, feature_set = _build_gnn_from_checkpoint(obj)
            return ("gnn", model, input_dim)
    except Exception:
        pass
    import joblib
    obj = joblib.load(MODEL_PATH)
    return ("sklearn", obj, None)


def _score_local(df: pd.DataFrame, model_meta) -> pd.DataFrame:
    """Score using local model (sklearn or GNN). model_meta is (kind, model, input_dim)."""
    kind, model, input_dim = model_meta
    if kind == "gnn":
        return _score_gnn(df, model, input_dim)
    # sklearn
    X = df[FEATURE_COLUMNS].fillna(0).astype(np.float64)
    risk_scores = model.predict_proba(X)[:, 1]
    risk_scores = np.clip(risk_scores, 0.0, 1.0)
    out = df.copy()
    out["risk_score"] = risk_scores
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        names = FEATURE_COLUMNS
        top_idx = np.argsort(imp)[::-1][:5]
        top_features = [(names[i], float(imp[i])) for i in top_idx]
        out["top_features"] = [top_features] * len(out)
    else:
        out["top_features"] = [[]] * len(out)
    return out


def _score_remote(df: pd.DataFrame) -> pd.DataFrame:
    """Score via MODEL_URL POST. Expects JSON body with rows, returns risk_scores and optional top_features."""
    import requests
    X = df[FEATURE_COLUMNS].fillna(0)
    payload = X.to_dict(orient="records")
    r = requests.post(
        MODEL_URL.rstrip("/") + "/predict",
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    risk_scores = np.array(data.get("risk_scores", data.get("scores", [])))
    risk_scores = np.clip(risk_scores, 0.0, 1.0)
    out = df.copy()
    out["risk_score"] = risk_scores
    out["top_features"] = data.get("top_features", [[]] * len(out))
    if len(out["top_features"]) != len(out):
        out["top_features"] = [[]] * len(out)
    return out


def score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run fraud model on preprocessed DataFrame.
    Uses MODEL_PATH (joblib or torch GNN checkpoint) if set, else MODEL_URL. Appends risk_score and top_features.
    """
    if MODEL_PATH:
        missing_gnn = [c for c in ["Account", "Account.1"] if c not in df.columns]
        if missing_gnn:
            raise ValueError(f"Missing columns for scoring: {missing_gnn}")
        missing_tab = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_tab:
            raise ValueError(f"Missing feature columns: {missing_tab}")

    if MODEL_PATH:
        model_meta = _get_model()
        return _score_local(df, model_meta)
    if MODEL_URL:
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return _score_remote(df)
    raise RuntimeError("Set MODEL_PATH or MODEL_URL to score transactions.")
