"""Test which backend works with your model."""
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 1)[0].rsplit('\\', 1)[0])

from pipeline.data import load_data
from laplace_model import laplace_model
import pymc as pm

# Load data
d = load_data()
d_dict = {
    'coords': d.coords,
    'all_formats': d.all_formats,
    'n_grids': d.n_grids,
    'n_stores': d.n_stores,
    'n_formats': d.n_formats,
    'pair_format_idx': d.pair_format_idx,
    'pair_brand_idx': d.pair_brand_idx,
    'log_walk': d.log_walk,
    'log_drive': d.log_drive,
    'log_surface_norm': d.log_surface_norm,
    'sidx': d.sidx,
    'gidx': d.gidx,
    'log_surface_our_fmt_norm': d.log_surface_our_fmt_norm,
    'log_store_pop_norm': d.log_store_pop_norm,
    'demo_norm': d.demo_norm,
    'demo_cols': d.demo_cols,
    'wealth_norm': d.wealth_norm,
    'our_stores_idx': d.our_stores_idx,
    'population_array': d.population_array,
    'observed_revenue': d.observed_revenue,
    'ancien_our_mois': d.ancien_our_mois,
    'alcool_our': d.alcool_our,
    'flat_gf_idx': d.flat_gf_idx,
    'pair_concept_idx': d.pair_concept_idx,
    'poi_types': d.poi_types,
    'log_pois': d.log_pois,
}

spec = {
    'likelihood': 'laplace',
    'prob_method': 'mnl',
}

print("=" * 60)
print("Testing NUMPY backend (baseline)")
print("=" * 60)
try:
    model_np = laplace_model(d_dict, spec)
    with model_np:
        ad_np = pm.ADVI()
        approx_np = ad_np.fit(n=100, backend="numpy")  # small sample for quick test
    print("✓ NUMPY backend works!")
except Exception as e:
    print(f"✗ NUMPY backend failed: {e}")

print("\n" + "=" * 60)
print("Testing JAX backend")
print("=" * 60)
try:
    model_jax = laplace_model(d_dict, spec)
    with model_jax:
        ad_jax = pm.ADVI()
        approx_jax = ad_jax.fit(n=100, backend="jax")  # small sample for quick test
    print("✓ JAX backend works!")
except Exception as e:
    print(f"✗ JAX backend failed:")
    print(f"  {type(e).__name__}: {str(e)[:200]}")
    import traceback
    traceback.print_exc()
