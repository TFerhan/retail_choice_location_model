"""GPU NUTS approach runner — Phase 1 & 2-3."""

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# GPU memory limits — MUST be set before any JAX import
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # don't hog GPU memory upfront
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]   = "platform" # release memory as soon as possible
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"    # optional, only if GPU is used

import pytensor
pytensor.config.floatX = "float32"

import arviz as az
import numpy as np
import pymc as pm

BASE_DIR = Path(__file__).parent.parent
OUTPUTS = BASE_DIR / "outputs"

DRAWS, TUNE, CHAINS = 50, 100, 2


@dataclass
class ApproachResult:
    name: str
    converged: bool
    sampler: str | None
    draws: int
    r_hat_max: float
    ess_min: float
    elapsed_s: float
    error: str | None = None


# def _data_bundle_to_dict(db) -> dict:
#     """Convert DataBundle to the dict format expected by build_model."""
#     return {
#         "n_grids": db.n_grids, "n_stores": db.n_stores, "N_pairs": db.N_pairs,
#         "n_formats": db.n_formats, "n_brands": db.n_brands,
#         "n_communes": db.n_communes, "n_villes": db.n_villes, "n_concepts": db.n_concepts,
#         "gidx": db.gidx, "sidx": db.sidx,
#         "pair_format_idx": db.pair_format_idx,
#         "pair_brand_idx": db.pair_brand_idx,
#         "pair_concept_idx": db.pair_concept_idx,
#         "our_stores_idx": db.our_stores_idx,
#         "shop_unique": db.shop_unique, "format_codes": db.format_codes, "all_formats": db.all_formats,
#         "brand_unique": db.brand_unique,
#         "concept_unique": db.concept_unique, "concept_codes": db.concept_codes,
#         "commune_codes": db.commune_codes, "commune_unique": db.commune_unique,
#         "ville_codes": db.ville_codes, "ville_unique": db.ville_unique,
#         "log_walk": db.log_walk, "log_drive": db.log_drive,
#         "population_array": db.population_array,
#         "wealth_norm": db.wealth_norm, "wealth_bayes_norm": db.wealth_bayes_norm,
#         "grid_unique": db.grid_unique,
#         "log_surface_norm": db.log_surface_norm,
#         "log_store_pop": db.log_store_pop, "log_store_pop_norm": db.log_store_pop_norm,
#         "pm_resid": db.pm_resid,
#         "days_norm": db.days_norm,
#         "rayon_array": db.rayon_array, "rayon_cols": db.rayon_cols,
#         "log_pois": db.log_pois, "pois_aligned": db.pois_aligned, "poi_types": db.poi_types,
#         "demo_matrix": db.demo_matrix, "demo_norm": db.demo_norm, "demo_cols": db.demo_cols,
#         "flat_gf_idx": db.flat_gf_idx,
#         "observed_sales_our": db.observed_sales_our,
#         "observed_revenue": db.observed_revenue,
#         "total_other": db.total_other,
#         "ancien_our": db.ancien_our,
#         "coords": db.coords,
#         "stores_indexed": db.stores_indexed,
#         "our_shop_codes": db.our_shop_codes,
#     }

def _data_bundle_to_dict(db) -> dict:
    """Convert DataBundle to the dict format expected by build_model."""
    return {
        # counts
        "n_grids": db.n_grids,
        "n_stores": db.n_stores,
        "N_pairs": db.N_pairs,
        "n_formats": db.n_formats,
        "n_brands": db.n_brands,
        "n_communes": db.n_communes,
        "n_concepts": db.n_concepts,

        # indices
        "gidx": db.gidx,
        "sidx": db.sidx,
        "pair_format_idx": db.pair_format_idx,
        "pair_brand_idx": db.pair_brand_idx,
        "pair_concept_idx": db.pair_concept_idx,
        "our_stores_idx": db.our_stores_idx,
        "our_store_format_idx": db.our_store_format_idx,
        "flat_gf_idx": db.flat_gf_idx,

        # store metadata
        "shop_unique": db.shop_unique,
        "format_codes": db.format_codes,
        "all_formats": db.all_formats,
        "brand_unique": db.brand_unique,
        "concept_unique": db.concept_unique,
        "concept_codes": db.concept_codes,
        "commune_codes": db.commune_codes,
        "commune_unique": db.commune_unique,

        # pair features
        "log_walk": db.log_walk,
        "log_drive": db.log_drive,

        # grid features
        "population_array": db.population_array,
        "wealth_norm": db.wealth_norm,
        "wealth_bayes_norm": db.wealth_bayes_norm,
        "wealth_norm_shifted": db.wealth_norm_shifted,
        "grid_unique": db.grid_unique,

        # store features
        "log_surface_norm": db.log_surface_norm,
        "log_surface_our_fmt_norm": db.log_surface_our_fmt_norm,
        "log_store_pop": db.log_store_pop,
        "log_store_pop_norm": db.log_store_pop_norm,
        "rayon_array": db.rayon_array,
        "rayon_cols": db.rayon_cols,

        # POIs
        "log_pois": db.log_pois,
        "pois_aligned": db.pois_aligned,
        "poi_types": db.poi_types,

        # demographics
        "demo_matrix": db.demo_matrix,
        "demo_norm": db.demo_norm,
        "demo_cols": db.demo_cols,

        # observations
        "observed_sales_our": db.observed_sales_our,
        "observed_revenue": db.observed_revenue,
        "total_other": db.total_other,

        # our-store covariates
        "ancien_our_mois": db.ancien_our_mois,
        "alcool_our": db.alcool_our,

        # misc
        "coords": db.coords,
        "stores_indexed": db.stores_indexed,
        "our_shop_codes": db.our_shop_codes,
    }


def _is_oom_error(e: Exception) -> bool:
    msg = str(e).upper()
    return "OUT OF MEMORY" in msg or "RESOURCE_EXHAUSTED" in msg or "OOM" in msg


def sample_approach(model, name: str):
    """Sample a PyMC model with GPU NUTS, falling back across samplers.

    Order: numpyro (vectorized) → nutpie.
    BlackJAX skipped — incompatible with PyMC 5.28.
    On OOM, retry once with halved draws.
    """
    samplers = [
        ("numpyro",  {"chain_method": "vectorized"}),
        ("nutpie",   {}),
    ]
    for draws in [DRAWS, DRAWS // 2]:
        for sampler, kwargs in samplers:
            try:
                with model:
                    idata = pm.sample(
                        nuts_sampler=sampler,
                        draws=draws, tune=TUNE, chains=CHAINS,
                        target_accept=0.85,
                        nuts_sampler_kwargs=kwargs,
                        idata_kwargs={"log_likelihood": False},
                        progressbar=True,
                    )
                idata.attrs["sampler"] = sampler
                idata.attrs["draws"] = draws
                return idata, sampler
            except Exception as e:
                print(f"  [{name}] {sampler} failed: {e}")
                if _is_oom_error(e):
                    print(f"  [{name}] OOM detected, will retry with halved draws ({draws // 2})")
                    break  # break inner loop to try halved draws
                traceback.print_exc(file=sys.stderr)
        else:
            break  # no OOM, all samplers failed at this draw count
    return None, None


def compute_convergence(idata) -> dict:
    """Compute r-hat max and ESS min from InferenceData."""
    try:
        summary = az.summary(idata, var_names=["~"], round_to=4)
    except Exception:
        return {"r_hat_max": 999.0, "ess_min": 0}

    r_hat_col = [c for c in summary.columns if "r_hat" in c]
    ess_bulk_col = [c for c in summary.columns if "ess_bulk" in c]

    r_hat_max = float(summary[r_hat_col[0]].max()) if r_hat_col else 999.0
    ess_min = float(summary[ess_bulk_col[0]].min()) if ess_bulk_col else 0.0

    return {"r_hat_max": r_hat_max, "ess_min": ess_min}


def run_approach(db, spec: dict) -> ApproachResult:
    """Build model, sample, save idata + metadata."""
    # Import build_model from run_selected without modifying it
    sys.path.insert(0, str(BASE_DIR))
    from run_selected import build_model

    name = spec["name"]
    d = _data_bundle_to_dict(db)
    out_dir = OUTPUTS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{name}] building model")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        print("here")
        model = build_model(d, spec)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{name}] build failed: {e}")
        return ApproachResult(
            name=name, converged=False, sampler=None,
            draws=0, r_hat_max=999, ess_min=0, elapsed_s=round(elapsed, 1),
            error=f"build: {e}",
        )

    idata, sampler = sample_approach(model, name)
    elapsed = time.time() - t0

    if idata is None:
        print(f"  [{name}] ALL samplers failed")
        return ApproachResult(
            name=name, converged=False, sampler=None,
            draws=0, r_hat_max=999, ess_min=0, elapsed_s=round(elapsed, 1),
            error="all samplers failed",
        )

    # Save idata
    idata.to_netcdf(out_dir / "idata.nc")
    print(f"  [{name}] saved idata ({sampler})")

    # Convergence
    conv = compute_convergence(idata)
    r_hat_max = conv["r_hat_max"]
    ess_min = conv["ess_min"]
    converged = r_hat_max <= 1.05

    if not converged:
        print(f"  WARNING: [{name}] r_hat_max={r_hat_max:.4f} > 1.05 — NOT CONVERGED")

    # Save metadata
    actual_draws = int(idata.attrs.get("draws", DRAWS))
    meta = {
        "approach": name,
        "converged": converged,
        "sampler": sampler,
        "draws": actual_draws,
        "tune": TUNE,
        "chains": CHAINS,
        "r_hat_max": round(r_hat_max, 4),
        "ess_min": round(ess_min, 1),
        "elapsed_s": round(elapsed, 1),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [{name}] done: r_hat={r_hat_max:.4f} ess={ess_min:.0f} time={elapsed:.0f}s")

    return ApproachResult(
        name=name, converged=converged, sampler=sampler,
        draws=actual_draws, r_hat_max=r_hat_max, ess_min=ess_min,
        elapsed_s=round(elapsed, 1),
    )


def run_all_approaches(db, approach_specs: list[dict] | None = None):
    """Run all approaches and return list of ApproachResult."""
    if approach_specs is None:
        sys.path.insert(0, str(BASE_DIR))
        from run_selected import APPROACHES
        approach_specs = APPROACHES

    results = []
    for i, spec in enumerate(approach_specs):
        print(f"\n>>> Approach {i+1}/{len(approach_specs)}: {spec['name']}")
        result = run_approach(db, spec)
        results.append(result)
    return results
