"""Load GNN, load pre-computed features, run full-graph inference, return scored df and account risk scores."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from app.config import MODEL_PATH, MODEL_DIR
from app.models.gnn_models import load_gnn_model

logger = logging.getLogger(__name__)

# Pre-computed feature/graph data directory
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = _BACKEND_DIR / "data" / "kaggle" / "working" / "processed_data"
FEATURE_DIR = PROCESSED_DATA_DIR / "features"
META_DIR = PROCESSED_DATA_DIR / "metadata"


def _load_precomputed(device: torch.device) -> dict:
    """Load pre-computed features (A+B), graph topology, and account maps."""
    # Load base graph data (edge_index, y, num_nodes)
    base_data = torch.load(META_DIR / "base_graph_data.pt", map_location=device, weights_only=False)
    edge_index = base_data.edge_index
    y = base_data.y

    # Load account mappings
    with open(META_DIR / "account_maps.pkl", "rb") as f:
        maps = pickle.load(f)
    account_to_id = maps["account_to_id"]
    id_to_account = maps["id_to_account"]

    # Load feature blocks A (behavioral, 54-dim) and B (random walk, 4-dim)
    X_a = torch.load(FEATURE_DIR / "features_behavioral_test.pt", weights_only=False)
    X_b = torch.load(FEATURE_DIR / "features_random_walk_test.pt", weights_only=False)
    X = torch.cat([X_a, X_b], dim=1)

    return {
        "X": X,
        "edge_index": edge_index,
        "y": y,
        "account_to_id": account_to_id,
        "id_to_account": id_to_account,
        "N": base_data.num_nodes,
    }


def run_gnn(
    df: pd.DataFrame,
    model_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Load GNN and pre-computed features, run full-graph inference, then map
    risk scores back to the accounts/transactions in df.
    Returns (scored_df with risk_score, account_risk_scores dict).
    """
    path = model_path or MODEL_PATH
    if not path or not str(path).strip():
        raise RuntimeError("MODEL_PATH is not set and no model_path provided")
    path = Path(path)
    if not path.is_absolute():
        candidate = MODEL_DIR / path.name
        path = candidate if candidate.exists() else path

    if not PROCESSED_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Pre-computed features not found at {PROCESSED_DATA_DIR}. "
            "Run the GNN training notebook first to generate them."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-computed features and graph
    logger.info("Loading pre-computed features from %s", PROCESSED_DATA_DIR)
    data = _load_precomputed(device)
    X = data["X"].to(device)
    edge_index = data["edge_index"].to(device)
    account_to_id = data["account_to_id"]
    id_to_account = data["id_to_account"]
    N = data["N"]

    logger.info("Loaded %d nodes, %d edges, feature dim=%d", N, edge_index.shape[1], X.shape[1])

    # Load model
    model, input_dim = load_gnn_model(path)
    assert X.shape[1] == input_dim, (
        f"Feature dim mismatch: pre-computed={X.shape[1]}, model expects={input_dim}"
    )

    # Run full-graph inference
    model.eval()
    with torch.no_grad():
        log_probs = model(X, edge_index)
    prob_pos = log_probs.exp()[:, 1].cpu().numpy()

    logger.info(
        "GNN inference done: risk_score range [%.4f, %.4f], flagged=%d (>=0.5)",
        prob_pos.min(), prob_pos.max(), (prob_pos >= 0.5).sum(),
    )

    # Build account_risk_scores: account_id (str) -> max risk score
    account_risk_scores: dict[str, float] = {}
    for node_idx in range(N):
        acc = id_to_account.get(node_idx)
        if acc is not None:
            acc_str = str(acc)
            score = float(prob_pos[node_idx])
            if acc_str not in account_risk_scores or score > account_risk_scores[acc_str]:
                account_risk_scores[acc_str] = score

    # Map each transaction row to the risk score of its "from" account
    src_col = "Account"
    out = df.copy()
    risk_scores = []
    for acc in df[src_col]:
        acc_str = str(acc)
        if acc_str in account_risk_scores:
            risk_scores.append(account_risk_scores[acc_str])
        else:
            # Account not in the pre-computed graph — default to 0
            risk_scores.append(0.0)

    out["risk_score"] = np.array(risk_scores, dtype=np.float64)
    out["top_features"] = [[]] * len(out)
    return out, account_risk_scores
