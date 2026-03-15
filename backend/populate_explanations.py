"""
Populate aiExplanation in nodes.csv for flagged accounts using Railtracks agents + Watsonx/Gemini.
Reads nodes.csv + edges.csv, runs 3-agent Railtracks analysis, then generates per-account explanations.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Load .env BEFORE importing app modules that read env vars
load_dotenv(BACKEND_DIR / ".env")

sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def _call_llm(system: str, user: str, max_retries: int = 3) -> str:
    """Call Gemini via LiteLLM with rate-limit retry."""
    import litellm
    for attempt in range(max_retries):
        try:
            r = litellm.completion(
                model="gemini/gemini-2.0-flash",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "RateLimitError" in err_str:
                wait = 35 * (attempt + 1)
                log.warning("Rate limited, waiting %ds before retry %d/%d...", wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                log.warning("LLM call failed: %s", e)
                break
    return "Automated explanation unavailable. Review flagged accounts manually."


DATA_DIR = PROJECT_ROOT / "frontend" / "public" / "data"
NODES_CSV = DATA_DIR / "nodes.csv"
EDGES_CSV = DATA_DIR / "edges.csv"

BATCH_SIZE = 25  # accounts per LLM call for per-account explanations


def main():
    df_nodes = pd.read_csv(NODES_CSV)
    df_edges = pd.read_csv(EDGES_CSV)
    log.info("Loaded %d nodes, %d edges", len(df_nodes), len(df_edges))

    # Set normal accounts first
    mask_normal = df_nodes["risk"] == "normal"
    df_nodes.loc[mask_normal, "aiExplanation"] = "No anomalies detected. Account activity within normal parameters."
    log.info("Set %d normal accounts to default explanation", mask_normal.sum())

    # Identify flagged accounts needing explanations
    flagged_mask = df_nodes["risk"].isin(["laundering", "suspicious"])
    flagged = df_nodes[flagged_mask].copy()
    log.info("Flagged accounts to explain: %d", len(flagged))

    if len(flagged) == 0:
        df_nodes.to_csv(NODES_CSV, index=False)
        log.info("No flagged accounts. Done.")
        return

    # Load existing explanations (resume support)
    existing = {}
    for _, r in df_nodes.iterrows():
        val = str(r.get("aiExplanation", ""))
        if val and val not in ("TOBEFILLED", "nan", "", "Automated explanation unavailable. Review flagged accounts manually."):
            existing[str(r["id"])] = val
    log.info("Loaded %d existing explanations from CSV (will skip those)", len(existing))

    # Build edge lookup for connection context
    edge_counts = {}
    for _, e in df_edges.iterrows():
        src, tgt = str(e["source"]), str(e["target"])
        edge_counts[src] = edge_counts.get(src, 0) + 1
        edge_counts[tgt] = edge_counts.get(tgt, 0) + 1

    # Step 1: Railtracks 3-agent analysis on top flagged accounts
    top_flagged = flagged.nlargest(50, "riskScore")
    summary_lines = []
    for _, row in top_flagged.iterrows():
        conns = edge_counts.get(str(row["id"]), 0)
        pat = row.get("pattern", "")
        if pd.isna(pat):
            pat = "None"
        summary_lines.append(
            f"Account: {row['id']} | Risk: {row['riskScore']:.3f} | "
            f"Risk Level: {row['risk']} | TxCount: {row['txCount']} | "
            f"Connections: {conns} | Patterns: {pat}"
        )
    summary = "\n".join(summary_lines)

    log.info("Running Railtracks Pattern Agent...")
    pattern_out = _call_llm(
        system="You are an AML pattern analyst. Given a list of flagged accounts with risk scores and transaction counts, summarize the laundering patterns observed. Focus on transaction volumes, connection patterns, and risk clustering. Be concise (3-5 sentences).",
        user=f"Flagged accounts:\n{summary}",
    )
    log.info("Pattern Agent done: %s", pattern_out[:100])

    log.info("Running Railtracks Risk Agent...")
    risk_out = _call_llm(
        system="You are an AML risk analyst. Given the same flagged accounts with risk scores and transaction data, comment on severity distribution, high-risk clusters, and which accounts warrant immediate investigation. Be concise (3-5 sentences).",
        user=f"Flagged accounts:\n{summary}",
    )
    log.info("Risk Agent done: %s", risk_out[:100])

    log.info("Running Railtracks Investigator Agent...")
    inv_out = _call_llm(
        system="You are an AML investigator. Given a pattern analyst summary and a risk analyst summary, write a brief overall assessment of the flagged network. 2-3 sentences.",
        user=f"Pattern analyst:\n{pattern_out}\n\nRisk analyst:\n{risk_out}",
    )
    log.info("Investigator Agent done: %s", inv_out[:100])

    # Step 2: Generate per-account explanations in batches via Watsonx/Gemini
    flagged_ids = flagged["id"].tolist()
    flagged_data = {str(r["id"]): r for _, r in flagged.iterrows()}
    explanations = dict(existing)  # Start with existing explanations

    for batch_start in range(0, len(flagged_ids), BATCH_SIZE):
        batch_ids = flagged_ids[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(flagged_ids) + BATCH_SIZE - 1) // BATCH_SIZE

        # Skip accounts that already have explanations (resume support)
        needs_work = [aid for aid in batch_ids if str(aid) not in explanations]
        if not needs_work:
            log.info("Batch %d/%d — all already explained, skipping", batch_num, total_batches)
            continue

        log.info("Generating explanations batch %d/%d (%d accounts)...", batch_num, total_batches, len(batch_ids))

        account_details = []
        for aid in batch_ids:
            row = flagged_data[str(aid)]
            conns = edge_counts.get(str(aid), 0)
            pat = row.get("pattern", "")
            if pd.isna(pat):
                pat = "None"
            account_details.append(
                f"- {aid}: risk_score={row['riskScore']:.4f}, risk_level={row['risk']}, "
                f"tx_count={row['txCount']}, connections={conns}, patterns={pat}"
            )

        prompt_system = """You are a fraud investigator assistant generating concise explanations for flagged accounts on an AML dashboard.

Context from network analysis:
PATTERN ANALYSIS: """ + pattern_out + """
RISK ANALYSIS: """ + risk_out + """

For each account below, write a unique 1-2 sentence explanation of why it was flagged, referencing its specific risk score, transaction volume, and any patterns. Make each explanation distinct and specific to that account's data.

Return ONLY a valid JSON object mapping account_id to explanation string. No markdown, no code fences."""

        prompt_user = "Accounts:\n" + "\n".join(account_details)

        raw = _call_llm(prompt_system, prompt_user)

        # Parse JSON response
        try:
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned.rsplit("```", 1)[0]
            batch_explanations = json.loads(cleaned)
            for aid in batch_ids:
                key = str(aid)
                if key in batch_explanations:
                    explanations[key] = batch_explanations[key]
                else:
                    # Try without leading zeros or other format variations
                    for k, v in batch_explanations.items():
                        if k.strip() == key:
                            explanations[key] = v
                            break
            log.info("  Parsed %d/%d explanations", len([x for x in batch_ids if str(x) in explanations]), len(batch_ids))
        except json.JSONDecodeError as e:
            log.warning("  JSON parse failed for batch %d: %s", batch_num, e)
            log.warning("  Raw response: %s", raw[:200])
            # Fallback: use investigator summary for this batch
            for aid in batch_ids:
                explanations[str(aid)] = inv_out

    # Step 3: Write explanations back to nodes.csv
    filled = 0
    for idx, row in df_nodes.iterrows():
        aid = str(row["id"])
        if aid in explanations:
            df_nodes.at[idx, "aiExplanation"] = explanations[aid]
            filled += 1

    df_nodes.to_csv(NODES_CSV, index=False)
    log.info("Updated %d flagged accounts with AI explanations", filled)
    log.info("Wrote %s", NODES_CSV)

    # Also copy to node_data if it exists
    node_data_dir = PROJECT_ROOT / "frontend" / "public" / "node_data"
    if node_data_dir.exists():
        df_nodes.to_csv(node_data_dir / "nodes.csv", index=False)
        log.info("Also updated %s", node_data_dir / "nodes.csv")

    # Summary
    remaining = ((df_nodes["risk"].isin(["laundering", "suspicious"])) &
                 (df_nodes["aiExplanation"].isin(["TOBEFILLED", "nan", ""]))).sum()
    log.info("Done! Remaining TOBEFILLED: %d", remaining)


if __name__ == "__main__":
    main()
