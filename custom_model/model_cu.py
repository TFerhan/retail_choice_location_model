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
        "b_drive_mu":   0.0,
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
        "b_surf_sig":   0.1,
    },
    "MARKET": {
        "base_u":       0.8,
        "b_walk_mu_mu": 0.8,
        "b_walk_sig":   0.5,
        "b_drive_mu":   0.3,
        "b_drive_sig":  0.05,
        "wealth":       0.4,
        "b_surf_sig":   0.1,
    },
    "HYPER": {
        "base_u":       1.5,
        "b_walk_mu_mu": 0.4,
        "b_walk_sig":   0.5,
        "b_drive_mu":   0.15,
        "b_drive_sig":  0.05,
        "wealth":       0.2,
        "b_surf_sig":   0.1,
    },
    "HYPERCASH":{
        "base_u":       1.5,
        "b_walk_mu_mu": 0.4,
        "b_walk_sig":   0.5,
        "b_drive_mu":   0.15,
        "b_drive_sig":  0.05,
        "wealth":       0.2,
        "b_surf_sig":   0.1,
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
    prob_method = spec.get("prob_method", "mnl")
    priors = spec.get("priors", "standard")
    coords           = d["coords"]
    fp               = build_format_priors(d["all_formats"])
    n_grids   = int(d["n_grids"])      # plain Python int
    n_stores  = int(d["n_stores"])     # plain Python int
    n_formats = int(d["n_formats"])    # plain Python int
    #prob_method     = spec.get("prob_method", "mnl")
        


    with pm.Model(coords=coords) as model:

        #base_u_raw = pm.Normal("base_u_raw", mu=fp["base_u_mu"], sigma=1.5, dims="format")
        base_u_raw = pm.Normal("base_u_raw", mu=fp.get("base_u"), sigma=1.5, dims="format")
        base_u = pm.Deterministic("base_u", base_u_raw - base_u_raw.mean(), dims="format")

        
        # b_walk = pm.Normal("b_walk", mu= b_walk_mu, sigma=fp.get("b_walk_sig"), dims="format")
        # b_walk = pm.TruncatedNormal("b_walk", mu=1, sigma=fp.get("b_walk_sig"), lower=0.0, dims="format")
        b_walk_mu = pm.Normal("b_walk_mu", mu=fp.get("b_walk_mu_mu"), sigma=0.1)
        b_walk_raw = pm.Normal("b_walk_raw", mu=b_walk_mu, sigma=fp.get("b_walk_sig"), dims="format")
        b_walk = pm.Deterministic("b_walk", pt.math.softplus(b_walk_raw))

        
        # b_drive = pm.HalfNormal("b_drive", mu=b_drive_mu,sigma=fp.get("b_drive_sig"), dims="format")
        # b_drive = pm.TruncatedNormal("b_drive", mu=b_walk_mu, sigma=fp.get("b_drive_sig"), lower=0.0, dims="format")
        b_drive_mu = pm.Normal("b_drive_mu", mu=fp.get("b_drive_mu"),sigma=0.1)
        b_drive_raw = pm.Normal("b_drive_raw", mu=b_drive_mu, sigma=fp.get("b_drive_sig"), dims="format")
        b_drive = pm.Deterministic("b_drive", pt.math.softplus(b_drive_raw))

        b_surf = pm.HalfNormal("b_surf", sigma=fp.get("b_surf_sig"), dims="format")
        b_surf_x_pop = pm.Normal("b_surf_x_pop", mu=0.1, sigma=0.15)

        b_alcool = pm.HalfNormal("b_alcool",sigma=0.02)

        b_wealth = pm.Normal("b_wealth", mu=fp.get("wealth"), sigma=0.4, dims="format")

        b_demo = pm.Normal("b_demo", mu=0, sigma=0.3, shape=len(d["demo_cols"]))
      
        # sigma_concept = pm.HalfNormal("sigma_concept", sigma=0.3)
        # concept_raw   = pm.Normal("concept_raw", mu=0, sigma=1, dims="concept")
        # concept_eff   = pm.Deterministic("concept_eff", concept_raw * sigma_concept, dims="concept")

        # beta_pop = pm.TruncatedNormal("beta_pop", mu=1.0, sigma=0.4, lower=0.0, upper=2.0)
        # beta_pop_wide = pm.Normal("beta_pop", mu=1.0, sigma=0.5)
        # beta_pop_tight = pm.TruncatedNormal("beta_pop", mu=1.0, sigma=0.15, lower=0.3, upper=1.5)
        beta_pop_raw = pm.Normal("beta_pop_raw", mu=0.0, sigma=1.5)
        beta_pop = pm.Deterministic("beta_pop", 2.0 * pm.math.invlogit(beta_pop_raw))

   
        b_old  = pm.HalfNormal("b_old", sigma=0.01)
        v = base_u[d["pair_format_idx"]]

        bw = b_walk[d["pair_format_idx"]]

        v  = v - bw * d["log_walk"]

        bd = b_drive[d["pair_format_idx"]] 
        v  = v - bd * d["log_drive"]
        bs = b_surf[d["pair_format_idx"]] 
        v += bs * d["log_surface_our_fmt_norm"][d["sidx"]] 

        v += b_surf_x_pop * d["log_surface_our_fmt_norm"][d["sidx"]] * d["log_store_pop_norm"][d["sidx"]]
   
        v += pt.dot(d["demo_norm"][d["commune_codes"][d["sidx"]], :], b_demo)


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

        M = (pop_g**beta_pop) * pm_grid[d["gidx"]] * pt.exp(poi_effect) * 360.0

        spend    = P * M
        rev      = pt.zeros(n_stores)
        rev      = pt.inc_subtensor(rev[d["sidx"]], spend)
        pred_our = rev[d["our_stores_idx"]]
 

        ancien   = pt.constant(d["ancien_our_mois"], dtype="float32")
        pred_our = pred_our * pt.exp(b_old * ancien)

        alcool = pt.constant(d["alcool_our"],dtype="float32")
        pred_our = pred_our * pt.exp(b_alcool * alcool)

        other_rev = pt.reshape(rev.sum() - pred_our.sum(), (1,))
        rh = pt.concatenate([pred_our, other_rev])
        # rh = pt.concatenate([pred_our, [(rev.sum() - pred_our.sum())]])
        rh = pt.maximum(rh, 1e-6)
        pm.Deterministic("predicted_revenue_our_stores", pred_our)


        if likelihood_type == "lognormal":
            sig = pm.HalfNormal("sigma", sigma=0.5)
            # pm.LogNormal("obs", mu=pt.log(rh), sigma=sig, observed=d["observed_revenue"]).astype(np.float32)
            pm.LogNormal("obs", mu=pt.log(rh), sigma=sig, observed=d["observed_revenue"])
        elif likelihood_type == "laplace":
            bs2 = pm.HalfNormal("b_scale", sigma=0.5)
            # pm.Laplace("obs", mu=pt.log(rh), b=bs2, observed=np.log(d["observed_revenue"])).astype(np.float32)
            pm.Laplace("obs", mu=pt.log(rh), b=bs2, observed=np.log(d["observed_revenue"]))
        elif likelihood_type == "student_t":
            sig = pm.HalfNormal("sigma", sigma=0.5)
            nu  = pm.Exponential("nu", 1.0 / 3.0)
            pm.StudentT("obs", nu=nu, mu=pt.log(rh), sigma=sig,
                        observed=np.log(d["observed_revenue"]))

        model._spec = spec
    
    return model









