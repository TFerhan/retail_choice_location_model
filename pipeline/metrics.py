"""Pure-function metrics — no side effects, no file I/O."""

import numpy as np
import pandas as pd

TOTAL_REVENUE = 41_000_000_000.0


def compute_holdout_metrics(pred_per_store: dict, observed_per_store: dict,
                            our_stores_idx: np.ndarray, format_codes: np.ndarray) -> dict:
    """Compute MAPE and coverage for holdout stores.

    pred_per_store: {store_idx_in_our: PredictionResult}
    observed_per_store: {store_idx_in_our: float}
    """
    errors = []
    coverages = []
    details = []

    for idx in sorted(pred_per_store):
        pr = pred_per_store[idx]
        obs = observed_per_store[idx]
        if obs <= 0:
            continue
        abs_err = abs(pr.mean - obs)
        pct_err = abs_err / obs * 100
        covered = obs >= pr.hdi_low and obs <= pr.hdi_high
        errors.append(pct_err)
        coverages.append(covered)
        details.append({
            "store_our_idx": int(idx),
            "observed": round(obs, 0),
            "predicted_p50": round(pr.p50, 0),
            "predicted_p10": round(pr.p10, 0),
            "predicted_p90": round(pr.p90, 0),
            "hdi_low": round(pr.hdi_low, 0),
            "hdi_high": round(pr.hdi_high, 0),
            "abs_error": round(abs_err, 0),
            "pct_error": round(pct_err, 2),
            "covered": bool(covered),
        })

    return {
        "mape": round(float(np.mean(errors)), 2) if errors else None,
        "coverage_94": round(float(np.mean(coverages)), 4) if coverages else None,
        "n_stores": len(errors),
        "details": details,
    }


def business_metrics(pred_samples: np.ndarray, competitor_threshold: float = 0.05) -> dict:
    """Store-level business metrics from prediction samples.

    competitor_threshold: market share threshold for decision signal.
    """
    mean = float(np.mean(pred_samples))
    p10 = float(np.percentile(pred_samples, 10))
    p50 = float(np.percentile(pred_samples, 50))
    p90 = float(np.percentile(pred_samples, 90))

    return {
        "revenue_upside": round((p90 - p50) / p50, 4) if p50 > 0 else None,
        "revenue_risk": round((p50 - p10) / p50, 4) if p50 > 0 else None,
        "market_share_p50": round(p50 / TOTAL_REVENUE, 6),
        "decision_signal": p10 > competitor_threshold * TOTAL_REVENUE,
    }


def aggregate_comparison(results: list, holdout_results: dict) -> pd.DataFrame:
    """Build comparison table from ApproachResult list and holdout metrics dict."""
    rows = []
    for r in results:
        row = {
            "approach": r.name,
            "converged": r.converged,
            "sampler": r.sampler or "none",
            "draws": r.draws,
            "r_hat_max": round(r.r_hat_max, 4),
            "ess_min": round(r.ess_min, 1),
            "mape_holdout": holdout_results.get(r.name, {}).get("mape", None),
            "coverage_94": holdout_results.get(r.name, {}).get("coverage_94", None),
            "elapsed_s": r.elapsed_s,
        }
        if not r.converged:
            row["converged"] = False
        rows.append(row)
    return pd.DataFrame(rows)
