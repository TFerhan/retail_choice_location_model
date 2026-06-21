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
        "b_walk_sig":   1.5,
        "b_drive_mu":   0.8,
        "b_drive_sig":  0.2,
        "wealth":      -0.5,
        "b_surf_sig":   0.2,
        "pm_mu": 3.0,
        "pm_sigma" : 0.19
    },
    "PROXI": {
        "base_u":      -0.3,
        "b_walk_mu_mu": 1.8,
        "b_walk_sig":   1.0,
        "b_drive_mu":   0.6,
        "b_drive_sig":  0.3,
        "wealth":      -0.1,
        "b_surf_sig":   0.15,
        "pm_mu": 4.0,
        "pm_sigma" : 0.26
    },
    "MARKET": {
        "base_u":       0.8,
        "b_walk_mu_mu": 0.8,
        "b_walk_sig":   0.5,
        "b_drive_mu":   0.3,
        "b_drive_sig":  0.5,
        "wealth":       0.4,
        "b_surf_sig":   0.1,
        "pm_mu": 4.5,
        "pm_sigma" : 0.4
    },
    "HYPER": {
        "base_u":       1.5,
        "b_walk_mu_mu": 0.4,
        "b_walk_sig":   0.5,
        "b_drive_mu":   0.15,
        "b_drive_sig":  0.2,
        "wealth":       0.2,
        "b_surf_sig":   0.05,
        "pm_mu": 4.8,
        "pm_sigma" : 0.43
    },
    "HYPERCASH":{
        "base_u":       1.5,
        "b_walk_mu_mu": 0.4,
        "b_walk_sig":   0.1,
        "b_drive_mu":   0.15,
        "b_drive_sig":  0.2,
        "wealth":       0.2,
        "b_surf_sig":   0.05,
        "pm_mu": 5,
        "pm_sigma" : 0.43
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
        "pm_mu": get("pm_mu"),
        "pm_sigma": get("pm_sigma"),
    }


def laplace_model(d: dict, spec: dict):
    likelihood_type = spec.get("likelihood", "lognormal")
    coords           = d["coords"]
    fp               = build_format_priors(d["all_formats"])
    n_grids   = int(d["n_grids"])      
    n_stores  = int(d["n_stores"])     
    n_formats = int(d["n_formats"])    
    prob_method     = spec.get("prob_method", "mnl")
    priors          = spec.get("priors", "standard")


    with pm.Model(coords=coords) as model:

        base_u_raw = pm.Normal("base_u_raw", mu=fp.get("base_u"), sigma=1.5, dims="format")
        base_u = pm.Deterministic("base_u", base_u_raw - base_u_raw.mean(), dims="format")

        b_walk = pm.HalfNormal("b_walk", sigma=fp.get("b_walk_sig"), dims="format")
  

        b_drive = pm.HalfNormal("b_drive", sigma=fp.get("b_drive_sig"), dims="format")


        b_surf = pm.HalfNormal("b_surf", sigma=fp.get("b_surf_sig"), dims="format")
        b_surf_x_pop = pm.Normal("b_surf_x_pop", mu=0.1, sigma=0.15)

        b_alcool = pm.HalfNormal("b_alcool",sigma=0.5)

  
        # b_wealth = pm.Normal("b_wealth", mu=fp.get("wealth"), sigma=0.4, dims="format")

        b_demo = pm.Normal("b_demo", mu=0, sigma=0.3, shape=len(d["demo_cols"]))

        sigma_concept = pm.HalfNormal("sigma_concept", sigma=0.3)
        concept_raw   = pm.Normal("concept_raw", mu=0, sigma=1, dims="concept")
        concept_eff   = pm.Deterministic("concept_eff", concept_raw * sigma_concept, dims="concept")

      
        beta_pop = pm.TruncatedNormal("beta_pop", mu=1.0, sigma=0.4, lower=0.0, upper=2.0)
  
        b_old  = pm.HalfNormal("b_old", sigma=0.08)

        v = base_u[d["pair_format_idx"]]

        bw = b_walk[d["pair_format_idx"]]

        v  = v - bw * d["log_walk"]

        bd = b_drive[d["pair_format_idx"]] 
        v  = v - bd * d["log_drive"]
        bs = b_surf[d["pair_format_idx"]] 

        v += bs * d["log_surface_our_fmt_norm"][d["sidx"]] 


        v += b_surf_x_pop * d["log_surface_our_fmt_norm"][d["sidx"]] * d["log_store_pop_norm"][d["sidx"]]
   
        v += pt.dot(d["demo_norm"][d["commune_codes"][d["sidx"]], :], b_demo)

        # v += concept_eff[d["pair_concept_idx"]]

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

        base_pm = pm.LogNormal("base_pm",mu=fp.get("pm_mu"),sigma=fp.get("pm_sigma"), dims="format")

        wealth_strength = pm.HalfNormal("wealth_strength", sigma=0.5)

        max_uplift = pm.Beta("max_uplift", alpha=2, beta=2)

        wealth_effect = pm.math.tanh(wealth_strength * d["wealth_norm"])
        wealth_effect = (wealth_effect + 1) / 2  

        pm_grid = base_pm[d["pair_format_idx"]] * (1 + max_uplift * wealth_effect[d["gidx"]])

        pop_g = d["population_array"][d["gidx"]]

        sigma_poi = pm.HalfNormal("sigma_poi", sigma=0.05)

        b_poi = pm.Normal("b_poi", mu=0, sigma=sigma_poi, shape=len(d["poi_types"]))

        poi_effect = pt.dot(d["log_pois"][d["gidx"], :], b_poi)
   

        M = (pop_g**beta_pop) * pm_grid * pt.exp(poi_effect) * 360.0


        spend    = P * M
        rev      = pt.zeros(n_stores)
        rev      = pt.inc_subtensor(rev[d["sidx"]], spend)
        pred_our = rev[d["our_stores_idx"]]

        ancien   = pt.constant(d["ancien_our_mois"], dtype="float32")
        pred_our = pred_our * pt.exp(b_old * ancien)

        alcool = pt.constant(d["alcool_our"],dtype="float32")
        pred_our = pred_our * pt.exp(b_alcool * alcool)


        rh = pt.concatenate([pred_our, [(rev.sum() - pred_our.sum())]])
        rh = pt.maximum(rh, 1e-6)
        pm.Deterministic("predicted_revenue_our_stores", pred_our)


        n_our = len(d["our_stores_idx"])
        store_fmt_idx = d["our_store_format_idx"] 

        #  if likelihood_type == "laplace":
        #      b_f = pm.HalfNormal("b_f", sigma=1.0, dims="format")
        #      b_other = pm.HalfNormal("b_other", sigma=1.0)

        #      b_store = b_f[store_fmt_idx]                         
        #      b_vec   = pt.concatenate([b_store, pt.atleast_1d(b_other)])

        #      pm.Laplace("obs", mu=pt.log(rh), b=b_vec,
        #                 observed=np.log(d["observed_revenue"]))

        #  elif likelihood_type == "student_t":
        #      sigma_f = pm.HalfNormal("sigma_f", sigma=1.0, dims="format")
        #      sigma_other = pm.HalfNormal("sigma_other", sigma=1.0)
        #      nu = pm.Exponential("nu", 1.0 / 3.0)

        #      sigma_store = sigma_f[store_fmt_idx]
        #      sigma_vec   = pt.concatenate([sigma_store, pt.atleast_1d(sigma_other)])

        #      pm.StudentT("obs", nu=nu, mu=pt.log(rh), sigma=sigma_vec,
        #                  observed=np.log(d["observed_revenue"]))

        if likelihood_type == "laplace":
            bs2 = pm.HalfNormal("b_scale", sigma=1)
            pm.Laplace("obs", mu=pt.log(rh), b=bs2, observed=np.log(d["observed_revenue"]))
        elif likelihood_type == "student_t":
            sig = pm.HalfNormal("sigma", sigma=1)
            nu  = pm.Exponential("nu", 1.0 / 3.0)
            pm.StudentT("obs", nu=nu, mu=pt.log(rh), sigma=sig, observed=np.log(d["observed_revenue"]))

        model._spec = spec
    
    return model









