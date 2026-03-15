"""Watsonx.ai client for generating investigation summaries. No caching here; caller uses DB cache.

Tries IBM Watsonx first; falls back to Gemini via LiteLLM when Watsonx is
unavailable (missing credentials, project misconfiguration, etc.).
"""

import logging
import os
import time
from typing import List, Tuple

import requests

from app.config import WATSONX_APIKEY, WATSONX_PROJECT_ID, WATSONX_URL, WATSONX_MODEL_ID

logger = logging.getLogger(__name__)
WATSONX_TIMEOUT = 30

# IAM token cache
_iam_token: str = ""
_iam_token_expiry: float = 0.0


def _get_iam_token() -> str:
    """Exchange WATSONX_APIKEY for an IAM Bearer token, with caching."""
    global _iam_token, _iam_token_expiry
    if _iam_token and time.time() < _iam_token_expiry - 60:
        return _iam_token

    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": WATSONX_APIKEY,
        },
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    _iam_token = body["access_token"]
    _iam_token_expiry = body.get("expiration", time.time() + 3600)
    return _iam_token


def _gemini_fallback(prompt: str) -> str:
    """Call Gemini via LiteLLM as fallback when Watsonx is unavailable."""
    import litellm
    resp = litellm.completion(
        model="gemini/gemini-3-flash-preview",
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()[:1500]


def _build_prompt(transaction_row: dict, top_features: List[Tuple[str, float]] = None) -> str:
    """Build the exact prompt from spec."""
    tx_id = transaction_row.get("transaction_id", "N/A")
    acc_id = transaction_row.get("account_id", "N/A")
    amount = transaction_row.get("amount", 0)
    timestamp = transaction_row.get("timestamp", "N/A")
    merchant = transaction_row.get("merchant", transaction_row.get("Account.1", "N/A"))
    location = transaction_row.get("location", str(transaction_row.get("From Bank", "N/A")))
    device = transaction_row.get("device", transaction_row.get("Payment Format", "N/A"))
    risk_score = transaction_row.get("risk_score", 0)

    if top_features:
        feature_list = "\n".join(f"{name}: {val:+.2f}" for name, val in top_features[:10])
    else:
        feature_list = "N/A"

    return f"""You are a fraud investigator assistant. Given the following transaction facts and model feature contributions, generate a 1-3 sentence investigative summary focusing on the most important reasons this transaction was flagged.

Transaction:
- transaction_id: {tx_id}
- account_id: {acc_id}
- amount: ${amount}
- timestamp: {timestamp}
- merchant: {merchant}
- location: {location}
- device: {device}
- risk_score: {risk_score}

Feature contributions (highest → lowest):
{feature_list}

Provide:
1) A short plain-language summary (1-3 sentences).
2) The top 3 reasons ranked.

Keep the response concise (truncate beyond 200 tokens)."""


def generate_summary(
    transaction_row: dict,
    top_features: List[Tuple[str, float]] = None,
) -> str:
    """
    Call watsonx.ai to generate 1-3 sentence investigator summary.
    Uses WATSONX_URL, WATSONX_APIKEY, WATSONX_PROJECT_ID. 5s timeout.
    Returns summary text or raises on error. Caller must cache in DB.
    """
    prompt = _build_prompt(transaction_row, top_features)

    # Try Watsonx first
    if WATSONX_APIKEY and WATSONX_PROJECT_ID:
        try:
            token = _get_iam_token()
            base = (WATSONX_URL or "https://us-south.ml.cloud.ibm.com").rstrip("/")
            url = f"{base}/ml/v1/text/generation?version=2024-05-31"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            payload = {
                "input": prompt,
                "model_id": WATSONX_MODEL_ID,
                "project_id": WATSONX_PROJECT_ID,
                "parameters": {"max_new_tokens": 200},
            }
            r = requests.post(url, json=payload, headers=headers, timeout=WATSONX_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            text = (data.get("results", [{}])[0].get("generated_text", "") or "").strip()
            return text[:1500]
        except Exception as e:
            logger.warning("Watsonx failed, falling back to Gemini: %s", e)

    # Fallback to Gemini
    try:
        return _gemini_fallback(prompt)
    except Exception as e:
        logger.exception("Both Watsonx and Gemini failed: %s", e)
        raise RuntimeError(f"AI summary generation failed: {e}") from e
