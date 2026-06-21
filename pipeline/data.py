"""DataBundle — single source of truth for all model approaches."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pytensor
pytensor.config.floatX = 'float64'

TOTAL_REVENUE = 41_000_000_000.0
FREQ_YEAR = 364
BASE_DIR = Path(__file__).parent.parent

GRIDS = "grids_data_29_04_26.parquet"
STORES = "stores_train_1.parquet"
POIS = "pois_data_29_04_26.parquet"
DISTANCES = "distance_grid_store_04_06.parquet"
COMMUNES = "communes_data_29_04_26.parquet"

DEP_TOT_COM_25 = np.float64(340912703410.9)
DEP_TOT_NONAL_COM_25 =  np.float64(63150748964.100006)
DEP_TOT_AL_COM_25 =  np.float64(277761954448.30005)


@dataclass(frozen=True)
class DataBundle:
    """Immutable bundle of all precomputed arrays needed by every approach."""
    # counts
    n_grids: int
    n_stores: int
    N_pairs: int
    n_formats: int
    n_brands: int
    n_communes: int
    # n_villes: int
    n_concepts: int

    # indices
    gidx: np.ndarray
    sidx: np.ndarray
    pair_format_idx: np.ndarray
    pair_brand_idx: np.ndarray
    pair_concept_idx: np.ndarray
    our_stores_idx: np.ndarray
    our_store_format_idx: np.ndarray
    flat_gf_idx: np.ndarray

    # coordinates
    all_formats: np.ndarray
    shop_unique: np.ndarray
    format_codes: np.ndarray
    brand_unique: np.ndarray
    concept_unique: np.ndarray
    concept_codes: np.ndarray
    commune_codes: np.ndarray
    commune_unique: np.ndarray
    # ville_codes: np.ndarray
    # ville_unique: np.ndarray
    coords: dict[str, np.ndarray]

    # features
    log_walk: np.ndarray
    log_drive: np.ndarray
    population_array: np.ndarray
    log_surface_norm: np.ndarray
    log_store_pop: np.ndarray
    log_store_pop_norm: np.ndarray
    # pm_resid: np.ndarray
    wealth_norm: np.ndarray
    wealth_bayes_norm: np.ndarray
    wealth_norm_shifted: np.ndarray
    # days_norm: np.ndarray
    rayon_array: np.ndarray
    rayon_cols: list[str]
    log_pois: np.ndarray
    pois_aligned: np.ndarray
    poi_types: list[str]
    demo_matrix: np.ndarray
    demo_norm: np.ndarray
    demo_cols: list[str]
    grid_unique: np.ndarray
    log_surface_our_fmt_norm: np.ndarray

    # observations
    observed_sales_our: np.ndarray
    observed_revenue: np.ndarray
    total_other: float
    #ancien_our: np.ndarray
    ancien_our_mois: np.ndarray
    alcool_our: np.ndarray

    # metadata
    stores_indexed: pd.DataFrame
    our_shop_codes: np.ndarray


def load_data() -> DataBundle:
    """Load and precompute all arrays from parquet files."""
    grids_df = pd.read_parquet(BASE_DIR / GRIDS)
    stores_df = pd.read_parquet(BASE_DIR / STORES)
    dist_df = pd.read_parquet(BASE_DIR / DISTANCES)
    pois_df = pd.read_parquet(BASE_DIR / POIS)
    commune_df = pd.read_parquet(BASE_DIR / COMMUNES)

    dist_df = dist_df[dist_df["Shop code"].isin(stores_df["Shop code"])]

    

    stores_df["Cluster"] = stores_df["Cluster"].str.strip()
    stores_indexed = stores_df.set_index("Shop code")

    our_shop_codes = stores_df[stores_df["ours_CA"] == True]["Shop code"].values
    notna_surface_codes = stores_df[stores_df["Surface"].notna()]["Shop code"].values
    real_surface = stores_indexed.loc[notna_surface_codes, "Surface"].values.astype(np.float64)
    

    valid_grids = grids_df[grids_df["DN"] > 0]["centroid_idx"].values
    dist_df = dist_df[dist_df["distance_m_walk"].notna()]
    dist_df = dist_df[dist_df["centroid_idx"].isin(valid_grids)]
    grids_df = grids_df[grids_df["centroid_idx"].isin(valid_grids)]

    grid_codes, grid_unique = pd.factorize(dist_df["centroid_idx"], sort=True)
    shop_codes_f, shop_unique = pd.factorize(dist_df["Shop code"], sort=True)
    n_grids = len(grid_unique)
    n_stores = len(shop_unique)
    N_pairs = len(dist_df)
    gidx = grid_codes.astype(np.int32)
    sidx = shop_codes_f.astype(np.int32)

    format_array = stores_indexed.loc[shop_unique, "Cluster"].values
    all_formats = np.unique(format_array)
    format_codes = pd.Categorical(format_array, categories=all_formats).codes.astype(np.int32)
    n_formats = len(all_formats)
    pair_format_idx = format_codes[sidx].astype(np.int32)
    our_stores_idx = np.where(np.isin(shop_unique, our_shop_codes))[0].astype(np.int32)

    our_store_format_idx = format_codes[our_stores_idx].astype(np.int32)


    # find which our codes are MISSING from dist_df entirely
    missing = set(our_shop_codes) - set(shop_unique)

    # observed_sales_our = stores_indexed.loc[our_shop_codes, "CA N"].values.astype(np.float64)
    # total_other = DEP_TOT_AL_COM_25 - observed_sales_our.sum()
    # observed_revenue = np.append(observed_sales_our, total_other)

    # Compute observed sales only for our stores that exist in shop_unique
    our_stores_in_pairs = shop_unique[our_stores_idx]
    observed_sales_our = stores_indexed.loc[our_stores_in_pairs, "CA N"].values.astype(np.float64)
    total_other = DEP_TOT_AL_COM_25 - observed_sales_our.sum()
    observed_revenue = np.append(observed_sales_our, total_other).astype(np.float64)
    
    # log_walk = np.log(np.clip(dist_df["distance_m_walk"].values, 1, None)).astype(np.float64)
    # log_drive = np.log(np.clip(dist_df["duration_s_drive"].values, 1, None)).astype(np.float64)

    log_walk = np.log(
    np.clip(dist_df["distance_m_walk"].values / 1000.0, 0.05, None)  
            ).astype(np.float64)

    log_drive = np.log(
            np.clip(dist_df["duration_s_drive"].values / 60.0, 1.0, None)    
        ).astype(np.float64)

    population_array = (
        grids_df.set_index("centroid_idx").loc[grid_unique, "DN"].values.astype(np.float64)
    )

    stores_indexed["log_surface"] = np.log(stores_indexed["Surface"].clip(lower=1e-6))



    format_stats = (
        stores_indexed
        .groupby("Cluster")["log_surface"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "log_mean", "std": "log_std"})
    )

    surface_fmt = stores_indexed.loc[notna_surface_codes, "Cluster"].values
    log_surface_real_raw = np.log(real_surface.clip(min=1e-6))   

    # log_surface_our_fmt_norm = np.zeros_like(log_surface_real_raw, dtype=np.float64)

    # for i, fmt in enumerate(surface_fmt):
    #     mean = format_stats.loc[fmt, "log_mean"]
    #     std = format_stats.loc[fmt, "log_std"]
    #     if std > 0:
    #         log_surface_our_fmt_norm[i] = (log_surface_real_raw[i] - mean) / std
    #     else:
    #         log_surface_our_fmt_norm[i] = 0.0

        # no longer needed: notna_surface_codes, real_surface, surface_fmt, etc.

    # surface_raw: all stores, keep it uncommented
    surface_raw = stores_indexed.loc[shop_unique, "Surface"].values.astype(float)

    # existing per-store normalized surface (used elsewhere)
    log_surface_raw = np.log(np.clip(surface_raw, 1e-6, None))
    surface_missing_mask = np.isnan(surface_raw)
    fmt = format_codes
    log_surface_norm = np.zeros_like(log_surface_raw, dtype=np.float64)
    for fi in range(n_formats):
        mask = (fmt == fi)
        mu = format_stats.loc[all_formats[fi], "log_mean"]
        std = format_stats.loc[all_formats[fi], "log_std"]
        if std <= 1e-8:
            std = 1.0
        obs = mask & (~surface_missing_mask)
        log_surface_norm[obs] = (log_surface_raw[obs] - mu) / std
        miss = mask & surface_missing_mask
        log_surface_norm[miss] = 0.0

    # NEW: full-length normalized surface for interaction term (store-level)
    log_surface_our_fmt_norm = np.zeros(n_stores, dtype=np.float64)
    for i in range(n_stores):
        s = surface_raw[i]
        if np.isnan(s):
            continue   # stays 0.0
        fmt_name = all_formats[format_codes[i]]
        mean = format_stats.loc[fmt_name, "log_mean"]
        std = format_stats.loc[fmt_name, "log_std"]
        if std > 0:
            log_surface_our_fmt_norm[i] = (np.log(max(s, 1e-6)) - mean) / std
    #pm_raw = stores_indexed.loc[shop_unique, "PM 2025"].fillna(np.nan).values.astype(float)
    # log_pm = np.log(np.clip(pm_raw, 1e-6, None))
    # log_pm_filled = np.copy(log_pm)
    # for fi in range(n_formats):
    #     mask = format_codes == fi
    #     fm = log_pm[mask][np.isfinite(log_pm[mask])]
    #     if len(fm) > 0:
    #         for ii in range(len(log_pm)):
    #             if format_codes[ii] == fi and not np.isfinite(log_pm[ii]):
    #                 log_pm_filled[ii] = fm.mean()
    # fmt_pm_means = np.array(
    #     [log_pm_filled[format_codes == fi].mean() for fi in range(n_formats)], dtype=np.float64
    # )
    # pm_resid = (log_pm_filled - fmt_pm_means[format_codes]).astype(np.float64)

    wealth_raw = (
        grids_df.set_index("centroid_idx")
        .loc[grid_unique, "Indice_richesse_IRIS"].fillna(0).values.astype(np.float64)
    )
    wealth_norm = ((wealth_raw - wealth_raw.mean()) / (wealth_raw.std() + 1e-8)).astype(np.float64)
    median_w = np.nanmedian(wealth_raw)
    wealth_bayes = np.where(np.isnan(wealth_raw), median_w, wealth_raw)
    wealth_bayes = np.where(wealth_bayes == 0, median_w, wealth_bayes)
    wealth_bayes_norm = (
        (wealth_bayes - wealth_bayes.mean()) / (wealth_bayes.std() + 1e-8)
    ).astype(np.float64)

    wealth_norm_shifted = ((wealth_raw - wealth_raw.mean() + wealth_raw.min()) / (wealth_raw.std() + 1e-8)).astype(np.float64)

    grids_commune = grids_df.set_index("centroid_idx")["code_commune"].loc[grid_unique].values
    store_commune_raw = np.zeros(n_stores, dtype=int)
    for si in range(n_stores):
        mask = sidx == si
        nearest_g = gidx[mask][np.argmin(log_walk[mask])]
        store_commune_raw[si] = int(grids_commune[nearest_g])
    commune_codes_vals, commune_unique = pd.factorize(store_commune_raw, sort=True)
    n_communes = len(commune_unique)
    commune_codes = commune_codes_vals.astype(np.int32)
    grid_commune_codes, grid_commune_unique = pd.factorize(grids_commune, sort=True)
    grid_commune_codes = grid_commune_codes.astype(np.int32)

    store_maxpop = np.zeros(n_stores, dtype=np.float64)
    for si in range(n_stores):
        mask = sidx == si
        store_maxpop[si] = float(population_array[gidx[mask]].max() + 1)
    log_store_pop = np.log(store_maxpop).astype(np.float64)
    log_store_pop_norm = (
        (log_store_pop - log_store_pop.mean()) / (log_store_pop.std() + 1e-8)
    ).astype(np.float64)

    # days_raw = stores_indexed.loc[shop_unique, "anciennete_2025"].fillna(300).values.astype(float)
    # days_norm = ((days_raw - days_raw.mean()) / (days_raw.std() + 1e-8)).astype(np.float64)

    months_raw = stores_indexed.loc[our_shop_codes, "months_passed_f"].values.astype(float)
    months_norm = ((months_raw - months_raw.mean()) / (months_raw.std() + 1e-8 )).astype(np.float64)

    brand_array = stores_indexed.loc[shop_unique, "Brand"].values
    brand_names, brand_unique = pd.factorize(brand_array, sort=True)
    n_brands = len(brand_unique)
    brand_names = brand_names.astype(np.int32)
    pair_brand_idx = brand_names[sidx].astype(np.int32)

    flat_gf_idx = (gidx * n_formats + pair_format_idx).astype(np.int32)

    pois_wide = pois_df.pivot(index="centroid_idx", columns="POI_type", values="count").fillna(0)
    pois_aligned = pois_wide.reindex(grid_unique, fill_value=0).values.astype(np.float64)
    log_pois = np.log1p(pois_aligned).astype(np.float64)
    poi_types = pois_wide.columns.tolist()

    # ville_array = stores_indexed.loc[shop_unique, "Ville"].values
    # ville_codes_vals, ville_unique = pd.factorize(ville_array, sort=True)
    # n_villes = len(ville_unique)
    # ville_codes = ville_codes_vals.astype(np.int32)

    concept_array = stores_indexed.loc[shop_unique, "Shop concept"].values
    concept_codes_vals, concept_unique = pd.factorize(concept_array, sort=True)
    n_concepts = len(concept_unique)
    concept_codes = concept_codes_vals.astype(np.int32)
    pair_concept_idx = concept_codes[sidx].astype(np.int32)

    # alcool_brand   = {'Atacadão', 'Hyper U', 'U Express', 'Carrefour Hyper',
    #    'Carrefour Market', 'Carrefour Express'}

    store_format_arr = stores_indexed.loc[shop_unique, "Cluster"].values
    store_brand_arr  = stores_indexed.loc[shop_unique, "Brand"].values
    is_ours          = np.isin(shop_unique, our_shop_codes)

    alcool_known_brands = {
    "Atacadão",
    "Hyper U",
    "U Express",
    "Carrefour Hyper",
    "Carrefour Market",
    "Carrefour Express",
}

    alcool_known_mask = (
        np.isin(store_brand_arr, list(alcool_known_brands))
        & (~is_ours)
    ).astype(np.float64)

    alcool_our = (
    stores_indexed.loc[
        shop_unique[our_stores_idx],
        "rayon_alcool"
    ]
    .fillna(False)
    .astype(np.float64)
    .values
    )

    # effective flag: 1.0 only where (known=True AND has alcool shelf)
    # for unknown stores: 0.0 → scale stays 1.0 regardless of b_alcool
    

    rayon_cols = [
        "rayon_alcool",
        # "Rayon poisson (rayon_poisson)",
        # "Rayon FLEG (rayon_fel)",
        # "Rayon Boucherie Volaille (rayon_boucherie_volaille)",
        # "Rayon traditionnel (rayon_traditionnel)",
        # "Rayon stand coup (rayon_stand_coup)",
    ]
    rayon_array = np.column_stack([
        stores_indexed.loc[shop_unique, c].fillna(False).astype(float).values
        for c in rayon_cols
    ]).astype(np.float64)

    alcool_effective = (rayon_array[:, 0] * alcool_known_mask).astype(np.float64)

    demo_cols = [
        "female_higher_education", "female_no_education",
        "avg_household_size", "homeowner_pct", "female_private_sector_employee",
    ]
    demo_matrix = np.zeros((n_communes, len(demo_cols)), dtype=np.float64)
    for ci in range(n_communes):
        ccode = commune_unique[ci]
        matches = commune_df[commune_df["code"] == ccode]
        if len(matches) > 0:
            for di, dc in enumerate(demo_cols):
                val = matches.iloc[0][dc]
                demo_matrix[ci, di] = val if np.isfinite(val) else 0.0
    demo_norm = np.zeros_like(demo_matrix)
    for di in range(demo_matrix.shape[1]):
        col = demo_matrix[:, di]
        if col.std() > 0:
            demo_norm[:, di] = (col - col.mean()) / col.std()

    # ancien_our = (
    #     stores_indexed.loc[shop_unique[our_stores_idx], "anciennete_2025"]
    #     .values.astype(np.float64) / 365.0
    # )

    # ancien_our_mois = (
    #     stores_indexed.loc[shop_unique[our_stores_idx], "months_elapsed_f"]
    #     .values.astype(np.float64) / 13.0
    # )

    ancien_our_mois = (
        stores_indexed.loc[shop_unique[our_stores_idx], "months_passed_f"]
        .values.astype(np.float64) / 13.0
    )

    coords = {
        "format": all_formats,
        "brand": brand_unique,
        "commune": commune_unique,
        # "ville": ville_unique,
        "concept": concept_unique,
    }

    return DataBundle(
        n_grids=n_grids, n_stores=n_stores, N_pairs=N_pairs,
        n_formats=n_formats, n_brands=n_brands,
        n_communes=n_communes, n_concepts=n_concepts,
        gidx=gidx, sidx=sidx,
        pair_format_idx=pair_format_idx,
        pair_brand_idx=pair_brand_idx,
        pair_concept_idx=pair_concept_idx,
        our_stores_idx=our_stores_idx,
        flat_gf_idx=flat_gf_idx,
        shop_unique=shop_unique, format_codes=format_codes, all_formats=all_formats,
        brand_unique=brand_unique,
        concept_unique=concept_unique, concept_codes=concept_codes,
        commune_codes=commune_codes, commune_unique=commune_unique,
        log_walk=log_walk, log_drive=log_drive,
        population_array=population_array,
        wealth_norm=wealth_norm, wealth_bayes_norm=wealth_bayes_norm, wealth_norm_shifted=wealth_norm_shifted,
        grid_unique=grid_unique,
        log_surface_norm=log_surface_norm,
        log_store_pop=log_store_pop, log_store_pop_norm=log_store_pop_norm,
        #pm_resid=pm_resid,
        #days_norm=days_norm,
        ancien_our_mois=ancien_our_mois,
        rayon_array=rayon_array, rayon_cols=rayon_cols,
        log_pois=log_pois, pois_aligned=pois_aligned, poi_types=poi_types,
        demo_matrix=demo_matrix, demo_norm=demo_norm, demo_cols=demo_cols,
        observed_sales_our=observed_sales_our,
        observed_revenue=observed_revenue,
        total_other=total_other,
        #ancien_our=ancien_our,
        coords=coords,
        stores_indexed=stores_indexed,
        our_shop_codes=our_shop_codes,
        log_surface_our_fmt_norm=log_surface_our_fmt_norm,
        alcool_our=alcool_our,
        our_store_format_idx=our_store_format_idx
        # ville_codes=ville_codes, ville_unique=ville_unique,
        # n_villes=n_villes,
    )

