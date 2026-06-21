"""Post-inference engine — pure numpy predictions from posterior draws."""

import json
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictionResult:
    mean: float
    p10: float
    p50: float
    p90: float
    hdi_low: float
    hdi_high: float
    samples: np.ndarray


def _build_utility(posterior_dict: dict, d: dict, spec: dict,
                   pair_slice: slice) -> tuple:
    """Build utility for a batch of pairs. Returns (v, ev, dn_contrib) for MNL.

    v: (n_draws, batch_size)
    ev: (n_draws, batch_size)  — exp(clipped v)
    """
    def g(n):
        v = posterior_dict.get(n)
        return np.atleast_1d(v) if v is not None else None

    bu = g("base_u")
    bw = g("b_walk")
    bd = g("b_drive")
    bs = g("b_surf")

    if bu.ndim == 1:
        bu = bu[:, None]
    if bw.ndim == 1:
        bw = bw[:, None]
    if bd.ndim == 1:
        bd = bd[:, None]
    if bs.ndim == 1:
        bs = bs[:, None]

    ft = spec.get("features", set())
    pf = d["pair_format_idx"][pair_slice]
    si = d["sidx"][pair_slice]

    v = np.take(bu, pf, axis=1)
    v -= np.take(bw, pf, axis=1) * d["log_walk"][pair_slice]
    v -= np.take(bd, pf, axis=1) * d["log_drive"][pair_slice]
    v += np.take(bs, pf, axis=1) * d["log_surface_norm"][si]

    if "surf_x_pop" in ft:
        bsp = g("b_surf_x_pop")
        if bsp is not None:
            v += bsp[:, None] * d["log_surface_norm"][si] * d["log_store_pop_norm"][si]
    if "pm_resid" in ft:
        bpm = g("b_pm_resid")
        if bpm is not None:
            v += bpm[:, None] * d["pm_resid"][si]
    if "commune_effect" in ft or "commune_demo" in ft:
        gam = g("gamma")
        if gam is not None:
            v += gam[:, d["commune_codes"][si]]
    if "ville_effect" in ft:
        ve = g("ville_eff")
        if ve is not None:
            v += ve[:, d["ville_codes"][si]]
    if "brand_effect" in ft:
        be = g("brand_eff")
        if be is not None:
            v += be[:, d["pair_brand_idx"][pair_slice]]
    if "concept_effect" in ft:
        ce = g("concept_eff")
        if ce is not None:
            v += ce[:, d["pair_concept_idx"][pair_slice]]
    if "rayon_effect" in ft:
        br = g("b_rayon")
        if br is not None:
            v += np.einsum("dn,nr->dr", br, d["rayon_array"][si])
    if "days_effect" in ft:
        bdd = g("b_days")
        if bdd is not None:
            v += bdd[:, None] * d["days_norm"][si]

    ev = np.exp(np.clip(v, -50, 50))
    return v, ev


def _numpy_predict(posterior_dict: dict, d: dict, spec: dict) -> np.ndarray:
    """Batched numpy prediction over all posterior draws.

    Returns array of shape (n_draws, n_our_stores).
    Processes pairs in batches to control memory.
    """
    def g(n):
        v = posterior_dict.get(n)
        return np.atleast_1d(v) if v is not None else None

    bu = g("base_u")
    if bu.ndim == 1:
        bu = bu[:, None]
    bp = g("beta_pop")
    pa = g("panier_moyen") or g("panier")
    bp = bp if bp.ndim > 1 else bp[:, None]
    pa = pa if pa.ndim > 1 else pa[:, None]

    n_draws = bu.shape[0]
    ft = spec.get("features", set())
    pm2 = spec.get("prob_method", "mnl")
    has_market_scale = spec.get("market_scale", False)
    has_fmt_freq = spec.get("fmt_freq", False)

    N = d["N_pairs"]
    BATCH = 200_000  # ~80MB per batch for utility (100 draws × 200K × 4B)

    # Accumulators: denominator per grid, revenue per store
    den = np.zeros((n_draws, d["n_grids"]), dtype=np.float32)

    if pm2 == "nested":
        lam = g("lambda")
        if lam.ndim == 1:
            lam = lam[:, None]
        seg_acc = np.zeros((n_draws, d["n_grids"] * d["n_formats"]), dtype=np.float32)

    # First pass: compute exp(V) and accumulate denominators
    # Store exp(V) and spend multipliers for second pass
    # To avoid storing all P, accumulate revenue directly
    rev = np.zeros((n_draws, d["n_stores"]), dtype=np.float32)

    # Precompute market size per pair
    pg = d["population_array"][d["gidx"]]
    if has_fmt_freq:
        lf = g("log_freq")
        pf_all = d["pair_format_idx"]
        freq_per_pair = np.exp(np.take(lf, pf_all, axis=1)) if lf is not None else 364
        M_raw = (pg ** bp[:, None]) * freq_per_pair * pa[:, None]
    else:
        M_raw = (pg ** bp[:, None]) * 364 * pa[:, None]
    if has_market_scale:
        mls = g("market_log_scale")
        if mls is not None:
            M_raw = M_raw * np.exp(mls[:, None])

    if pm2 == "nested":
        # Nested logit: two-stage probability
        # First accumulate seg sums, then compute P
        ev_all = np.zeros((n_draws, N), dtype=np.float32)
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            sl = slice(start, end)
            _, ev_batch = _build_utility(posterior_dict, d, spec, sl)
            ev_all[:, start:end] = ev_batch
            # Accumulate segment sums
            np.add.at(seg_acc, d["flat_gf_idx"][sl], ev_batch)
            # Accumulate denominators
            np.add.at(den, d["gidx"][sl], ev_batch)

        # Compute nested probabilities
        sm = seg_acc.reshape(n_draws, d["n_grids"], d["n_formats"])
        Ig = np.log(sm + 1e-10)
        nu = bu[:, None, :] + lam[:, None, :] * Ig
        if "wealth_nest" in ft:
            bwl = g("b_wealth")
            if bwl is not None:
                w = d["wealth_bayes_norm"] if "wealth_bayes" in ft else d["wealth_norm"]
                nu += bwl[:, None, :] * w[None, :, None]
        nf = np.concatenate([nu, np.zeros((n_draws, d["n_grids"], 1), dtype=np.float32)], axis=2)
        en = np.exp(np.clip(nf, -50, 50))
        Pn = (en / en.sum(axis=2, keepdims=True))[:, :, :d["n_formats"]]

        # Second pass: compute P and revenue in batches
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            sl = slice(start, end)
            ev_b = ev_all[:, start:end]
            Pw = ev_b / (sm[:, d["gidx"][sl], d["pair_format_idx"][sl]] + 1e-10)
            Pn_b = Pn[:, d["gidx"][sl], d["pair_format_idx"][sl]]
            P = Pw * Pn_b
            spend = P * M_raw[:, start:end]
            np.add.at(rev, d["sidx"][sl], spend)
        del ev_all
    else:
        # MNL: two passes — accumulate denominators, then compute revenue
        # Store ev per batch to avoid recomputing utilities
        n_batches = (N + BATCH - 1) // BATCH
        ev_batches = [None] * n_batches

        for batch_i in range(n_batches):
            start = batch_i * BATCH
            end = min(start + BATCH, N)
            sl = slice(start, end)
            _, ev_batch = _build_utility(posterior_dict, d, spec, sl)
            ev_batches[batch_i] = ev_batch
            np.add.at(den, d["gidx"][sl], ev_batch)

        den += 1.0
        for batch_i in range(n_batches):
            start = batch_i * BATCH
            end = min(start + BATCH, N)
            sl = slice(start, end)
            ev_batch = ev_batches[batch_i]
            P = ev_batch / den[:, d["gidx"][sl]]
            spend = P * M_raw[:, start:end]
            np.add.at(rev, d["sidx"][sl], spend)
            ev_batches[batch_i] = None  # free memory

    pred = rev[:, d["our_stores_idx"]]

    if "age_effect" in ft:
        bo = g("b_old")
        if bo is not None:
            pred *= 1.0 + bo[:, None] * d["ancien_our"][None, :]

    return pred


@dataclass
class PostInference:
    """Post-inference wrapper around fitted posterior."""
    idata: object  # arviz InferenceData
    data_dict: dict
    spec: dict

    def fit(self):
        """Store posterior draws for prediction."""
        posterior = self.idata.posterior
        # Flatten chain+draw dims into single sample dim
        self.posterior_dict = {}
        for var_name in posterior:
            arr = posterior[var_name].values  # (chain, draw, ...)
            leading = int(np.prod(arr.shape[:2]))
            self.posterior_dict[var_name] = arr.reshape(leading, -1)

        # If shape is (n_samples, 1), squeeze to (n_samples,)
        for var_name in self.posterior_dict:
            arr = self.posterior_dict[var_name]
            if arr.ndim == 2 and arr.shape[1] == 1:
                self.posterior_dict[var_name] = arr.squeeze(axis=1)

        self.n_samples = list(self.posterior_dict.values())[0].shape[0]

    def predict_all(self) -> np.ndarray:
        """Predict revenue for all our stores across all posterior draws."""
        return _numpy_predict(self.posterior_dict, self.data_dict, self.spec)

    def predict_store(self, store_idx: int) -> PredictionResult:
        """Predict revenue for a single store (by our_stores_idx position)."""
        pred = self.predict_all()
        samples = pred[:, store_idx]
        return _to_prediction_result(samples)

    def summary(self) -> pd.DataFrame:
        """P10, P50, P90, mean, HDI 94% for each store."""
        pred = self.predict_all()
        rows = []
        for i in range(pred.shape[1]):
            pr = _to_prediction_result(pred[:, i])
            rows.append({
                "store_idx": i,
                "mean": pr.mean, "p10": pr.p10, "p50": pr.p50, "p90": pr.p90,
                "hdi_low": pr.hdi_low, "hdi_high": pr.hdi_high,
            })
        return pd.DataFrame(rows)


def _to_prediction_result(samples: np.ndarray) -> PredictionResult:
    """Convert sample array to PredictionResult with quantiles and HDI."""
    hdi = az.hdi(samples, hdi_prob=0.94)
    return PredictionResult(
        mean=float(np.mean(samples)),
        p10=float(np.percentile(samples, 10)),
        p50=float(np.percentile(samples, 50)),
        p90=float(np.percentile(samples, 90)),
        hdi_low=float(hdi.min()),
        hdi_high=float(hdi.max()),
        samples=samples.copy(),
    )
