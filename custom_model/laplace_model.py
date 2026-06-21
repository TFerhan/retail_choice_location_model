import argparse
import json
import os
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

warnings.filterwarnings("ignore")

TOTAL_REVENUE = 41_000_000_000.0
FREQ_YEAR = 364
BASELINE_SPENDING = 30

# DEP_TOT_COM_25 = np.float64(340912703410.9)
# DEP_TOT_NONAL_COM_25 =  np.float64(63150748964.100006)
# DEP_TOT_AL_COM_25 =  np.float64(277761954448.30005)

laplace_like = {
        "name": "32_laplace",
        "features": set(),
        "priors": "standard",
        "likelihood": "laplace",
        "prob_method": "mnl",
    },

FORMAT_PRIORS = {
    "DISCOUNTER": {
        "base_u":      -1.0,   
        "b_walk_mu_mu": 2.5,
        "b_walk_sig":   0.5,
        "b_drive_mu":   0.8,
        "b_drive_sig":  0.05,
        "wealth":      -0.5,
        "b_surf_sig":   0.0125,
    },
    "PROXI": {
        "base_u":      -0.3,
        "b_walk_mu_mu": 1.8,
        "b_walk_sig":   0.5,
        "b_drive_mu":   0.6,
        "b_drive_sig":  0.05,
        "wealth":      -0.1,
        "b_surf_sig":   0.025,
    },
    "MARKET": {
        "base_u":       0.8,
        "b_walk_mu_mu": 0.8,
        "b_walk_sig":   0.5,
        "b_drive_mu":   0.3,
        "b_drive_sig":  0.05,
        "wealth":       0.4,
        "b_surf_sig":   0.00625,
    },
    "HYPER": {
        "base_u":       1.5,
        "b_walk_mu_mu": 0.4,
        "b_walk_sig":   0.5,
        "b_drive_mu":   0.15,
        "b_drive_sig":  0.05,
        "wealth":       0.2,
        "b_surf_sig":   0.00025,
    },
    "HYPERCASH":{
        "base_u":       1.5,
        "b_walk_mu_mu": 0.4,
        "b_walk_sig":   0.5,
        "b_drive_mu":   0.15,
        "b_drive_sig":  0.05,
        "wealth":       0.2,
        "b_surf_sig":   0.00025,
    },
    "Outside": {}
}

DEFAULT_FORMAT_PRIOR = {
    "base_u":    0.0,
    "wealth":    0.0,
    "surf_mean": 300,
}

def build_format_priors(formats):

    def get(key):
        return np.array([
            FORMAT_PRIORS.get(f, DEFAULT_FORMAT_PRIOR).get(key, 0.0)
            for f in formats
        ], dtype=np.float32)

    return {
        "base_u": get("base_u"),
        "wealth": get("wealth"),
        "b_walk_mu_mu": get("b_walk_mu_mu"),
        "b_walk_sig": get("b_walk_sig"),
        "b_drive_mu": get("b_drive_mu"),
        "b_drive_sig": get("b_drive_sig"),
        "b_surf_sig": get("b_surf_sig"),
    }


def laplace_model(d: dict, spec: dict):
    likelihood_type = spec.get("likelihood", "lognormal")
    coords           = d["coords"]
    fp               = build_format_priors(d["all_formats"])
    n_grids   = int(d["n_grids"])      # plain Python int
    n_stores  = int(d["n_stores"])     # plain Python int
    n_formats = int(d["n_formats"])    # plain Python int
    prob_method     = spec.get("prob_method", "mnl")
    priors          = spec.get("priors", "standard")


    with pm.Model(coords=coords) as model:

        #base_u_raw = pm.Normal("base_u_raw", mu=fp["base_u_mu"], sigma=1.5, dims="format")
        base_u_raw = pm.Normal("base_u_raw", mu=fp.get("base_u"), sigma=1.5, dims="format")
        base_u = pm.Deterministic("base_u", base_u_raw - base_u_raw.mean(), dims="format")

        b_walk = pm.HalfNormal("b_walk", sigma=fp.get("b_walk_sig"), dims="format")
        # b_walk_mu = pm.Normal("b_walk_mu", mu=fp.get("b_walk_mu_mu"), sigma=0.1)
        # b_walk = pm.Normal("b_walk", mu= b_walk_mu, sigma=fp.get("b_walk_sig"), dims="format")
        # b_walk = pm.TruncatedNormal("b_walk", mu=1, sigma=fp.get("b_walk_sig"), lower=0.0, dims="format")
        # b_walk_raw = pm.Normal("b_walk_raw", mu=0, sigma=fp.get("b_walk_sig"), dims="format")
        # b_walk = pm.Deterministic("b_walk", pm.math.softplus(b_walk_raw + b_walk_mu))

        b_drive = pm.HalfNormal("b_drive", sigma=fp.get("b_drive_sig"), dims="format")
        # b_drive_mu = pm.Normal("b_drive_mu", mu=fp.get("b_drive_mu"), sigma=0.1)
        # b_drive = pm.Normal("b_drive", mu=b_drive_mu,sigma=fp.get("b_drive_sig"), dims="format")
        # b_drive = pm.TruncatedNormal("b_drive", mu=b_walk_mu, sigma=fp.get("b_drive_sig"), lower=0.0, dims="format")
        # b_drive_raw = pm.Normal("b_drive_raw", mu=0, sigma=fp.get("b_drive_sig"), dims="format")
        # b_walk = pm.Deterministic("b_drive", pm.math.softplus(b_walk_raw + b_walk_mu))

        #b_surf must not be normal as surface influence a lot revenue , half normal ?
        b_surf = pm.HalfNormal("b_surf", sigma=fp.get("b_surf_sig"), dims="format")
        b_surf_x_pop = pm.Normal("b_surf_x_pop", mu=0.1, sigma=0.15)

        b_alcool = pm.HalfNormal("b_alcool",sigma=0.02)

        #b_surf_lambda = pm.Beta("b_surf_lambda", alpha=2, beta=2)

        #b_pm = pm.Normal("b_pm_resid",   mu=0.1, sigma=0.2)

        #b_loyalty = pm.HalfNormal("b_loyalty", sigma=0.1)


        #maybe
        # b_wealth = pm.Normal("b_wealth", mu=fp.get("wealth"), sigma=0.4, dims="format")


        # sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=0.3)
        # gamma_raw   = pm.Normal("gamma_raw", mu=0, sigma=1, dims="commune")
        # gamma       = pm.Deterministic("gamma", gamma_raw * sigma_gamma, dims="commune")

        b_demo = pm.Normal("b_demo", mu=0, sigma=0.3, shape=len(d["demo_cols"]))

        # sigma_ville = pm.HalfNormal("sigma_ville", sigma=0.5)
        # ville_raw   = pm.Normal("ville_raw", mu=0, sigma=1, dims="ville")
        # ville_eff   = pm.Deterministic("ville_eff", ville_raw * sigma_ville, dims="ville")

        sigma_concept = pm.HalfNormal("sigma_concept", sigma=0.3)
        concept_raw   = pm.Normal("concept_raw", mu=0, sigma=1, dims="concept")
        concept_eff   = pm.Deterministic("concept_eff", concept_raw * sigma_concept, dims="concept")

        #seperate pois of types
        #b_poi = pm.Normal("b_poi", mu=0, sigma=0.1, shape=len(d["poi_types"]))

        # b_rayon = pm.HalfNormal("b_rayon", mu=0, sigma=0.2, shape=len(d["rayon_cols"]))

        beta_pop = pm.TruncatedNormal("beta_pop", mu=1.0, sigma=0.4, lower=0.0, upper=2.0)
        # beta_pop_wide = pm.Normal("beta_pop", mu=1.0, sigma=0.5)
        # beta_pop_tight = pm.TruncatedNormal("beta_pop", mu=1.0, sigma=0.15, lower=0.3, upper=1.5)
        # beta_pop_raw = pm.Normal("beta_pop_raw", mu=0.0, sigma=1.5)
        # beta_pop = pm.Deterministic("beta_pop", 2.0 * pm.math.invlogit(beta_pop_raw))

        #panier_fmt = pm.LogNormal("panier_fmt", mu=np.log(50.0), sigma=0.3, dims="format")
        #panier = pm.LogNormal("panier_moyen", mu=np.log(50.0), sigma=0.5)

        #b_days = pm.Normal("b_days", mu=0.3, sigma=0.2)

        # maybe per format ? to be checked !! if it increase in the real data or no
        # b_old  = pm.HalfNormal("b_old",  mu=0.0, sigma=0.01, dims="format")

        b_old  = pm.HalfNormal("b_old", sigma=0.01)

        #this must be counts ? or no need ? it will be embeded in panier moyen per grid ?
        # freq = pm.HalfNormal("freq", sigma=50, dims="format")
        # log_freq = pm.Normal("log_freq", mu=np.log([104, 52, 156]), sigma=1.0, dims="format")

        #market_log_scale = pm.Normal("market_log_scale", mu=-2.0, sigma=1.0)

        v = base_u[d["pair_format_idx"]]

        bw = b_walk[d["pair_format_idx"]]

        #why negative? or let the bw learn it? or keep negative to force it
        v  = v - bw * d["log_walk"]

        bd = b_drive[d["pair_format_idx"]] 
        v  = v - bd * d["log_drive"]
        bs = b_surf[d["pair_format_idx"]] 
        #v  = v + bs * d["log_surface_norm"][d["sidx"]]

        v += bs * d["log_surface_our_fmt_norm"][d["sidx"]] 


        v += b_surf_x_pop * d["log_surface_our_fmt_norm"][d["sidx"]] * d["log_store_pop_norm"][d["sidx"]]
        
        # you cant add pm_resid, what if you want to predict another store revenue ? you wont have its panier moyen, as panier moyen is something after the store is done no?
        #v += b_pm * d["pm_resid"][d["sidx"]]


        #v += gamma[d["commune_codes"][d["sidx"]]]
   
        v += pt.dot(d["demo_norm"][d["commune_codes"][d["sidx"]], :], b_demo)

        #v += ville_eff[d["ville_codes"][d["sidx"]]]

        #v += concept_eff[d["pair_concept_idx"]]

        # v += pt.dot(d["rayon_array"][d["sidx"], :], b_rayon)

        # could be influencing, but you dont have it for all stores and also outside option
        #v += b_days * d["days_norm"][d["sidx"]]

        # should we add loyalty knowing that loyalty is common just between three format
        #v += b_loyalty * d["loyalty"]

        # ── Choice probabilities ─────────────────────────────────
        if prob_method == "nested":
            lam = pm.Beta("lambda", alpha=6.0 if priors == "informative" else 2.0, beta=2.0, dims="format")
            ev      = pt.exp(pt.clip(v, -50, 50))
            seg     = pt.zeros(n_grids * n_formats)
            seg     = pt.inc_subtensor(seg[d["flat_gf_idx"]], ev)
            seg_mat = seg.reshape((n_grids, n_formats))
            Ig      = pt.log(seg_mat + 1e-10)
            nu      = base_u[None, :] + lam[None, :] * Ig
            nf  = pt.concatenate([nu, pt.zeros((n_grids, 1))], axis=1)
            en  = pt.exp(pt.clip(nf, -50, 50))
            Pn  = (en / en.sum(axis=1, keepdims=True))[:, :n_formats]
            Pw  = ev / (seg_mat[d["gidx"], d["pair_format_idx"]] + 1e-10)
            P   = Pw * Pn[d["gidx"], d["pair_format_idx"]]
        else:
            ev  = pt.exp(pt.clip(v, -50, 50))
            den = pt.zeros(n_grids)
            den = pt.inc_subtensor(den[d["gidx"]], ev)
            den += 1.0
            P   = ev / den[d["gidx"]]

        base_pm = pm.LogNormal(
            "base_pm",
            mu=np.log(BASELINE_SPENDING),
            sigma=0.25
        )

        wealth_strength = pm.HalfNormal(
            "wealth_strength",
            sigma=0.5
        )

        max_uplift = pm.Beta(
            "max_uplift",
            alpha=2,
            beta=2
        )

        wealth_effect = pm.math.tanh(wealth_strength * d["wealth_norm"])
        wealth_effect = (wealth_effect + 1) / 2  

        pm_grid = base_pm *(1 + max_uplift * wealth_effect)

        pop_g = d["population_array"][d["gidx"]]

        sigma_poi = pm.HalfNormal("sigma_poi", sigma=0.05)

        b_poi = pm.Normal(
            "b_poi",
            mu=0,
            sigma=sigma_poi,
            shape=len(d["poi_types"])
        )

        poi_effect = pt.dot(
            d["log_pois"][d["gidx"], :],
            b_poi
        )
        # M = (pop_g**beta_pop) * pm_grid * pt.exp(poi_effect)

        M = (pop_g**beta_pop) * pm_grid[d["gidx"]] * pt.exp(poi_effect) * 360.0

        #M = M * pt.exp(market_log_scale)

        spend    = P * M
        rev      = pt.zeros(n_stores)
        rev      = pt.inc_subtensor(rev[d["sidx"]], spend)
        pred_our = rev[d["our_stores_idx"]]
        #ancien   = pt.constant(d["ancien_our"], dtype="float32")
        #pred_our = pred_our * (1.0 + b_old * ancien)

        # our_fmt_idx = pt.constant(d["format_codes"][d["our_stores_idx"]], dtype="int32")

        # # Age effect: b_old[format] * ancien (per store)
        # pred_our = pred_our * (b_old[our_fmt_idx] * pt.constant(d["ancien_our_mois"], dtype="float32"))

        ancien   = pt.constant(d["ancien_our_mois"], dtype="float32")
        pred_our = pred_our * pt.exp(b_old * ancien)

        alcool = pt.constant(d["alcool_our"],dtype="float32")
        pred_our = pred_our * pt.exp(b_alcool * alcool)

        

        
        # rh = pt.concatenate([pred_our, [(rev.sum() - pred_our.sum())]])
        # other_revenue = rev.sum() - pred_our.sum()
        # rh = pt.concatenate([pred_our, pt.shape_padright(other_revenue, 1)])
        # other_revenue_1d = pt.reshape(other_revenue, (1,))
        # rh = pt.concatenate([pred_our, other_revenue_1d])

        rh = pt.concatenate([pred_our, [(rev.sum() - pred_our.sum())]])
        rh = pt.maximum(rh, 1e-6)
        pm.Deterministic("predicted_revenue_our_stores", pred_our)


        if likelihood_type == "lognormal":
            sig = pm.HalfNormal("sigma", sigma=0.5)
            # pm.LogNormal("obs", mu=pt.log(rh), sigma=sig, observed=d["observed_revenue"]).astype(np.float32)
            pm.LogNormal("obs", mu=pt.log(rh), sigma=sig, observed=d["observed_revenue"])
        # elif likelihood_type == "normal_log":
        #     sig = pm.HalfNormal("sigma", sigma=1.0)
        #     pm.Normal("obs", mu=pt.log(rh), sigma=sig, observed=np.log(d["observed_revenue"]))
        # elif likelihood_type == "weighted":
        #     sig = pm.HalfNormal("sigma", sigma=0.5)
        #     nw  = len(d["our_stores_idx"])
        #     w   = np.ones(nw + 1, dtype=np.float32)
        #     w[:nw] = 5.0
        #     lo  = pt.constant(np.log(d["observed_revenue"]).astype(np.float32))
        #     pm.Potential("wll", (-0.5 * w * ((pt.log(rh) - lo) / sig) ** 2).sum())
        # elif likelihood_type == "hetero":
        #     so  = pm.HalfNormal("sigma_our",   sigma=0.3)
        #     st  = pm.HalfNormal("sigma_other", sigma=1.0)
        #     nw  = len(d["our_stores_idx"])
        #     lo  = pt.constant(np.log(d["observed_revenue"]).astype(np.float32))
        #     lp  = pt.log(rh)
        #     pm.Potential(
        #         "hll",
        #         (-0.5 * ((lp[:nw] - lo[:nw]) / so) ** 2).sum()
        #         + (-0.5 * ((lp[nw:] - lo[nw:]) / st) ** 2).sum(),
        #     )
        # elif likelihood_type == "huber":
        #     sig = pm.HalfNormal("sigma", sigma=0.5)
        #     lo  = pt.constant(np.log(d["observed_revenue"]).astype(np.float32))
        #     r   = (pt.log(rh) - lo) / sig
        #     h   = pt.switch(pt.abs(r) < 1.35, 0.5 * r**2, 1.35 * pt.abs(r) - 0.91125)
        #     pm.Potential("hll", -h.sum())
        elif likelihood_type == "laplace":
            bs2 = pm.HalfNormal("b_scale", sigma=0.5)
            # pm.Laplace("obs", mu=pt.log(rh), b=bs2, observed=np.log(d["observed_revenue"])).astype(np.float32)
            pm.Laplace("obs", mu=pt.log(rh), b=bs2, observed=np.log(d["observed_revenue"]))
        elif likelihood_type == "student_t":
            sig = pm.HalfNormal("sigma", sigma=0.5)
            nu  = pm.Exponential("nu", 1.0 / 3.0)
            pm.StudentT("obs", nu=nu, mu=pt.log(rh), sigma=sig, observed=np.log(d["observed_revenue"]))

        model._spec = spec
    
    return model










