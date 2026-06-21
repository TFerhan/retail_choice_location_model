"""Main entry point: python -m pipeline.runner --phase N"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
OUTPUTS = BASE_DIR / "outputs"

# GPU memory limits — MUST be set before any JAX import
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.28"

import pytensor
pytensor.config.floatX = "float32"


def phase_0():
    """Environment check."""
    print("=== Phase 0: Environment Check ===")

    import jax
    print(f"JAX version: {jax.__version__}")
    devs = jax.devices("gpu")
    print(f"GPU devices: {devs}")

    import jax.numpy as jnp
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (100, 100))
    y = jnp.dot(x, x.T)
    print(f"GPU compute test: sum={float(jax.device_get(y.sum())):.1f}")

    import pymc
    import numpyro
    import blackjax
    print(f"PyMC: {pymc.__version__}")
    print(f"NumPyro: {numpyro.__version__}")
    print(f"BlackJAX: {blackjax.__version__}")
    try:
        import nutpie
        print(f"Nutpie: {nutpie.__version__}")
    except ImportError:
        print("Nutpie: not installed")

    from pipeline.data import load_data
    db = load_data()
    print(f"DataBundle loaded: {db.n_grids} grids, {db.n_stores} stores, {db.N_pairs} pairs")
    print("Phase 0: OK")
    return db


def phase_1(db):
    """Run all approaches with GPU NUTS. Skips approaches with valid idata."""
    print("\n=== Phase 1: GPU NUTS Sampling ===")
    sys.path.insert(0, str(BASE_DIR))
    from run_selected import APPROACHES

    from pipeline.runner import run_approach, ApproachResult
    results = []
    for i, spec in enumerate(APPROACHES):
        name = spec["name"]
        idata_path = OUTPUTS / name / "idata.nc"
        meta_path = OUTPUTS / name / "meta.json"
        if idata_path.exists() and meta_path.exists():
            print(f"\n>>> Approach {i+1}/{len(APPROACHES)}: {name} — already done, skipping")
            with open(meta_path) as f:
                m = json.load(f)
            results.append(ApproachResult(
                name=name, converged=m["converged"], sampler=m["sampler"],
                draws=m["draws"], r_hat_max=m["r_hat_max"], ess_min=m["ess_min"],
                elapsed_s=m["elapsed_s"],
            ))
            continue
        print(f"\n>>> Approach {i+1}/{len(APPROACHES)}: {name}")
        result = run_approach(db, spec)
        results.append(result)

    # Save summary
    summary = []
    for r in results:
        summary.append({
            "approach": r.name, "converged": r.converged, "sampler": r.sampler,
            "r_hat_max": r.r_hat_max, "ess_min": r.ess_min, "elapsed_s": r.elapsed_s,
        })
    with open(OUTPUTS / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    n_conv = sum(1 for r in results if r.converged)
    print(f"\nPhase 1 complete: {n_conv}/{len(results)} converged")
    return results


def select_holdout(db):
    """Select one holdout store per format (smallest observed revenue in that format)."""
    holdout_idx = {}
    for fi in range(db.n_formats):
        # Find our stores of this format
        format_our = []
        for j, oi in enumerate(db.our_stores_idx):
            if db.format_codes[oi] == fi:
                format_our.append((j, db.observed_sales_our[j]))
        if len(format_our) >= 3:
            # Choose the one with smallest observed revenue
            format_our.sort(key=lambda x: x[1])
            holdout_idx[fi] = format_our[0][0]  # store index in our_stores ordering

    return holdout_idx


def phase_2_3(db, approach_results):
    """Post-inference + holdout validation."""
    print("\n=== Phase 2-3: Post-Inference + Holdout ===")
    sys.path.insert(0, str(BASE_DIR))
    from run_selected import APPROACHES

    from pipeline.runner import _data_bundle_to_dict
    from pipeline.post_inference import PostInference
    from pipeline.metrics import compute_holdout_metrics

    holdout_idx = select_holdout(db)
    print(f"Holdout stores (format -> our_idx): {holdout_idx}")

    # Save holdout indices for each approach (same across all)
    for spec in APPROACHES:
        out_dir = OUTPUTS / spec["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "holdout_idx.json", "w") as f:
            json.dump(holdout_idx, f, indent=2)

    holdout_results = {}

    for spec in APPROACHES:
        name = spec["name"]
        idata_path = OUTPUTS / name / "idata.nc"
        if not idata_path.exists():
            print(f"  [{name}] no idata.nc, skipping post-inference")
            continue

        try:
            import arviz as az
            idata = az.from_netcdf(idata_path)
            d = _data_bundle_to_dict(db)

            pi = PostInference(idata=idata, data_dict=d, spec=spec)
            pi.fit()

            # Predict holdout stores
            pred_holdout = {}
            obs_holdout = {}
            for fi, store_our_idx in holdout_idx.items():
                pr = pi.predict_store(store_our_idx)
                pred_holdout[store_our_idx] = pr
                obs_holdout[store_our_idx] = db.observed_sales_our[store_our_idx]

            metrics = compute_holdout_metrics(
                pred_holdout, obs_holdout, db.our_stores_idx, db.format_codes
            )
            holdout_results[name] = metrics

            # Save holdout results
            saveable = dict(metrics)
            with open(OUTPUTS / name / "holdout_results.json", "w") as f:
                json.dump(saveable, f, indent=2)

            # Save normalizer (feature stats for predict_point)
            norm = {
                "log_surface_norm_mean": float(db.log_surface_raw_mean) if hasattr(db, 'log_surface_raw_mean') else 0,
                "n_our_stores": len(db.our_stores_idx),
            }
            with open(OUTPUTS / name / "normalizer.json", "w") as f:
                json.dump(norm, f, indent=2)

            mape = metrics.get("mape")
            cov = metrics.get("coverage_94")
            print(f"  [{name}] MAPE={mape}% coverage={cov}")

        except Exception as e:
            print(f"  [{name}] post-inference error: {e}")
            import traceback
            traceback.print_exc()
            holdout_results[name] = {"error": str(e)}

    return holdout_results


def phase_4(approach_results, holdout_results):
    """Build comparison CSV."""
    print("\n=== Phase 4: Comparison Table ===")
    from pipeline.metrics import aggregate_comparison

    comp_df = aggregate_comparison(approach_results, holdout_results)
    comp_df.to_csv(OUTPUTS / "comparison.csv", index=False)
    print(comp_df.to_string(index=False))
    return comp_df


def phase_5(comp_df, holdout_results):
    """Generate LaTeX report."""
    print("\n=== Phase 5: LaTeX Report ===")
    from pipeline.latex import generate_report

    out = generate_report(comp_df, holdout_results, BASE_DIR / "report" / "model_report.tex")
    print(f"Report written to {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Retail MNL GPU NUTS Pipeline")
    ap.add_argument("--phase", type=str, default="all",
                    help="Phase to run: 0, 1, 2-3, 4, 5, or all")
    ap.add_argument("--approach", type=str, help="Run single approach by name (phases 1-3 only)")
    args = ap.parse_args()

    if args.phase in ("0", "all"):
        db = phase_0()
    else:
        from pipeline.data import load_data
        db = load_data()

    if args.approach:
        # Single approach debug mode
        sys.path.insert(0, str(BASE_DIR))
        from run_selected import get_model_spec
        spec = get_model_spec(args.approach)

        from pipeline.runner import run_approach
        result = run_approach(db, spec)
        approach_results = [result]

        # Also do post-inference
        holdout_results = phase_2_3(db, approach_results)
        comp_df = phase_4(approach_results, holdout_results)
        phase_5(comp_df, holdout_results)
        return

    if args.phase in ("1", "all"):
        approach_results = phase_1(db)

    if args.phase in ("2-3", "all"):
        holdout_results = phase_2_3(db, approach_results)

    if args.phase in ("4", "all"):
        comp_df = phase_4(approach_results, holdout_results)

    if args.phase in ("5", "all"):
        phase_5(comp_df, holdout_results)

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
