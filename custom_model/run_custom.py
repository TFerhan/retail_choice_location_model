import json, os, sys, time, traceback, pytensor
import arviz as az
import numpy as np
import pymc as pm
from pathlib import Path
from pymc.sampling.jax import sample_numpyro_nuts  # = partial(sample_jax_nuts, nuts_sampler="numpyro")
from typing import Any
from run_selected import APPROACHES, build_model
from pipeline.runner import _is_oom_error, compute_convergence  # ← remove run_approach here
from pipeline.data import load_data, filter_formats
from laplace_model import laplace_model
from pipeline.runner import _data_bundle_to_dict, ApproachResult
import pymc_extras.inference.pathfinder as pf_mod

os.environ["JAX_PLATFORMS"]                = "cuda"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

orig_floatX = pytensor.config.floatX
pytensor.config.floatX = "float64"

BASE_DIR = Path(".")
OUTPUTS  = BASE_DIR / "outputs"
db       = load_data()

DRAWS  = 20
TUNE   = 5
CHAINS = 2

approach = APPROACHES[3]

name = approach["name"]
idata_path = OUTPUTS / name / "idata.nc"
meta_path = OUTPUTS / name / "meta.json"

cores_ = os.cpu_count() - 2


# full run
#d_full = vars(db)

# 2-format run
#d_two = filter_formats(bundle, keep_formats=["Discount", "Proximité"])

# 3-format run  
#d_three = filter_formats(bundle, keep_formats=["Discount", "Proximité", "Supermarché"])



def _pathfinder_initvals(model, name, n_chains, num_paths=None):
    num_paths = num_paths or max(n_chains * 2, 4)

    try:
        print(f"  [{name}] Running PathFinder ({num_paths} paths)...")
        try:
            with model:
              pf_datatree = pf_mod.fit_pathfinder(
                  num_paths=num_paths,
                  num_draws=n_chains,
                  num_draws_per_path=50,
                  importance_sampling="psis",
                  progressbar=True,
                  concurrent="process",          
                  cores=cores_,            
                  add_pathfinder_groups=False,
                  display_summary=False,
              )
        finally:
            pytensor.config.floatX = orig_floatX   

        print(f"  [{name}] PathFinder done")

        posterior   = pf_datatree["posterior"].ds
        n_pf_chains = posterior.sizes["chain"]
        n_pf_draws  = posterior.sizes["draw"]
        n_total     = n_pf_chains * n_pf_draws

        untrans_to_value = {
            rv.name: val.name
            for rv, val in model.rvs_to_values.items()
        }
        mappable = {
            pf_var: untrans_to_value[pf_var]
            for pf_var in posterior.data_vars
            if pf_var in untrans_to_value
        }
        if not mappable:
            print(f"  [{name}] PathFinder: no variable names matched — skipping init")
            return None, None

        replace  = n_total < n_chains
        idxs     = np.random.choice(n_total, size=n_chains, replace=replace)
        initvals = []
        for idx in idxs:
            ci = int(idx) // n_pf_draws
            di = int(idx) % n_pf_draws
            initvals.append({
                val_name: posterior[pf_var].values[ci, di]
                for pf_var, val_name in mappable.items()
            })

        pf_mean = np.concatenate([
            posterior[pf_var].values
                .reshape(-1, *posterior[pf_var].shape[2:])
                .mean(axis=0).ravel()
            for pf_var in mappable
        ])

        print(
            f"  [{name}] PathFinder: {n_total} draws → "
            f"{n_chains} inits ({'w/' if replace else 'w/o'} replacement)"
        )
        return initvals, pf_mean

    except Exception as e:
        print(f"  [{name}] PathFinder failed ({e}) — using default init")
        traceback.print_exc(file=sys.stderr)
        return None, None




def sample_approach(model, name: str):
    for draws in [DRAWS, DRAWS // 2]:
        initvals, _ = _pathfinder_initvals(model, name, n_chains=CHAINS)
        effective_tune = max(TUNE // 2, 200) if initvals is not None else TUNE

        try:
            with model:
                idata = sample_numpyro_nuts(
                    draws=draws,
                    tune=effective_tune,
                    chains=CHAINS,
                    target_accept=0.9,
                    chain_method="vectorized",        # ← top-level param, works correctly
                    nuts_kwargs={"dense_mass": False}, # ← valid NUTS kernel kwarg
                    initvals=initvals,                 # ← None is fine if PathFinder failed
                    idata_kwargs={"log_likelihood": False},
                    compute_convergence_checks=False,
                    progressbar=True,
                )

            idata.attrs["sampler"]         = "numpyro"
            idata.attrs["draws"]           = draws
            idata.attrs["tune"]            = effective_tune
            idata.attrs["pathfinder_init"] = initvals is not None
            return idata, "numpyro"

        except Exception as e:
            print(f"  [{name}] numpyro failed: {e}")
            if _is_oom_error(e):
                print(f"  [{name}] OOM → retrying with {draws // 2} draws")
                continue
            traceback.print_exc(file=sys.stderr)
            break

    return None, None


def run_approach(db, spec: dict):
    sys.path.insert(0, str(BASE_DIR))
    name    = spec["name"]
    d       = _data_bundle_to_dict(db)
    out_dir = OUTPUTS / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n[{name}] building model\n{'='*60}")
    t0 = time.time()
    try:
        model = build_model(d, spec)
    except Exception as e:
        print(f"  [{name}] build failed: {e}")
        return ApproachResult(
            name=name, converged=False, sampler=None,
            draws=0, r_hat_max=999, ess_min=0,
            elapsed_s=round(time.time() - t0, 1), error=f"build: {e}",
        )

    idata, sampler = sample_approach(model, name)
    elapsed = time.time() - t0

    if idata is None:
        print(f"  [{name}] ALL samplers failed")
        return ApproachResult(
            name=name, converged=False, sampler=None,
            draws=0, r_hat_max=999, ess_min=0,
            elapsed_s=round(elapsed, 1), error="all samplers failed",
        )

    idata.to_netcdf(out_dir / "idata.nc")
    print(f"  [{name}] saved idata ({sampler})")

    conv      = compute_convergence(idata)
    r_hat_max = conv["r_hat_max"]
    ess_min   = conv["ess_min"]
    converged = r_hat_max <= 1.05

    if not converged:
        print(f"  WARNING: [{name}] r_hat_max={r_hat_max:.4f} > 1.05 — NOT CONVERGED")

    actual_draws = int(idata.attrs.get("draws", DRAWS))
    meta = {
        "approach": name, "converged": converged, "sampler": sampler,
        "draws": actual_draws, "tune": TUNE, "chains": CHAINS,
        "r_hat_max": round(r_hat_max, 4), "ess_min": round(ess_min, 1),
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

print(f"\n>>> Approach: {name}")
result = run_approach(db, approach)
