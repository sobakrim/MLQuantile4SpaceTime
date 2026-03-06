#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic study for mlq4s (inspired by your Gapfilling EASY experiment).

This script:
  1) Builds a synthetic spatio-temporal dataset with covariate-driven non-Gaussian marginals:
        X(t) -> mu(t,s), sigma(t,s)
        Z(t,s) ~ latent spatio-temporal GRF (Gneiting)
        Y(t,s) = exp(mu(t,s) + sigma(t,s) * Z(t,s))
  2) Splits into train/test
  3) Builds an "oracle" ensemble on test (using the *true* mu/sigma + true GRF params)
  4) Fits mlq4s for several marginal methods (knn/qrf/qrnn) and simulates ensembles
  5) Produces paper-grade diagnostics:
        - time-series bands (oracle + each method) at a few sites
        - pooled QQ vs oracle
        - temporal ACF of latent Z (truth vs methods)
        - spatial correlation binned (truth vs methods)
  6) Saves results (npz + png figures + config text) to an output folder.

Assumptions:
  - Your package exposes:
        from mlq4s.pipeline import mlq4sModel
  - Your merged GRF file exposes:
        from mlq4s.st_grf import simulate_gneiting_jax

If your module names differ, adjust the imports below.

Run:
  python synthetic_study.py

"""

from __future__ import annotations

import os
import sys
import json
import logging
import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.special import kv as besselk, gamma
from scipy.linalg import toeplitz
from numpy.linalg import cholesky


# ============================================================
# Imports from your package (adjust if needed)
# ============================================================
try:
    from mlq4s.pipeline import mlq4sModel
    from mlq4s.st_grf import simulate_gneiting_jax
except Exception as e:
    raise ImportError(
        "Could not import mlq4sModel / simulate_gneiting_jax.\n"
        "Adjust the imports near the top of this script to match your package layout.\n"
        f"Original import error: {e}"
    )


# ============================================================
# Config
# ============================================================
@dataclass
class SyntheticStudyConfig:
    # Paths
    out_dir: str = "./outputs/synthetic_study"

    # Spatial grid
    nsx: int = 15
    nsy: int = 15

    # Time
    nt: int = 6000
    n_test: int = 200
    start_date: str = "2000-01-01"
    freq: str = "D"

    # Latent simulation budget
    L: int = 50_000
    chunk_size: int = 500

    # True latent Gneiting params: [a, kappa, nu, alpha, tau, q]
    pars_Z: Tuple[float, float, float, float, float, float] = (0.7, 3.4, 1.20, 0.3, 0.90, 0.70)

    # Covariates
    P: int = 10
    X1_kernel: Tuple[float, float, float, float] = (0.5, 0.5, 0.2, 1.0)  # a, alpha, tau, sigma
    X2_kernel: Tuple[float, float, float, float] = (0.1, 0.7, 0.2, 1.0)
    seed_X1: int = 32
    seed_X2: int = 2

    # Spatial coefficient fields for mu/sigma
    beta0: float = 1.0
    gamma0: float = -1.0
    b1_sigma2: float = 1e-1
    b1_rho: float = 0.8
    b1_nu: float = 1.5
    b1_seed: int = 10
    g1_sigma2: float = 1e-2
    g1_rho: float = 0.6
    g1_nu: float = 1.6
    g1_seed: int = 11

    # Ensembles
    n_sims_oracle: int = 50
    n_sims_per_method: int = 50

    # Quantile grid for marginals
    taus_n: int = 100

    # Plotting
    sites_to_plot: Tuple[int, int, int] = (0, 112, 224)  # for n=225
    max_lag_acf: int = 60
    spatial_bins: int = 18

    # Fitting knobs (kept close to your experiment)
    gneiting_block_size: int = 100
    gneiting_n_blocks: int = 100
    gneiting_t_max: int = 10
    gneiting_epsilon: float = 1e-3
    random_state: int = 0

    # Methods (you can edit these)
    # Note: these are passed as marginal_kwargs to mlq4sModel
    # (SitewiseMarginal reads standardize_X, kernel/k/h for knn, qrf_* for qrf, qrnn_* for qrnn)
    methods: Tuple[Tuple[str, Dict[str, Any]], ...] = (
        ("qrf", dict(
            standardize_X=False,
            qrf_select_hyperparams=False,
            n_estimators=2000,
            min_samples_leaf=50,
            max_features="log2",
            max_depth=None,
            bootstrap=True,
            min_samples_split=10,
            random_state=42,
            n_jobs=30,
            n_jobs_sites=30,
        )),
        ("knn", dict(
            standardize_X=False,
            kernel="gaussian",
            k=100,
            h=("adaptive", 0.5),
            n_jobs_sites=30,
        )),
        ("qrnn", dict(
            standardize_X=False,
            qrnn_select_hyperparams=False,
            n_layers=1,
            width=32,
            n_epochs=200,
            batch_size=128,
            device="cpu",
            n_jobs_sites=30,
        )),
    )


# ============================================================
# Logging
# ============================================================
def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"synthetic_study_{now}.log"

    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    logger = logging.getLogger("synthetic_study")
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("Logging to %s", log_file)
    return logger


# ============================================================
# Helpers (inspired from your experiment)
# ============================================================
def matern_covariance(dist, sigma2=1.0, rho=1.0, nu=1.5):
    dist = np.asarray(dist, float)
    s = np.sqrt(2.0 * nu) * dist / float(rho)
    core = np.zeros_like(s)
    mask = s > 0
    core[mask] = (s[mask] ** nu) * besselk(nu, s[mask])
    const = (2.0 ** (1.0 - nu)) / gamma(nu)
    C = sigma2 * const * core
    np.fill_diagonal(C, sigma2)
    return C


def simulate_matern_coeff_matrix(
    coords, n_realizations, *, sigma2=1e-4, rho=0.3, nu=1.5,
    nugget=1e-10, seed=None
):
    rng = np.random.default_rng(seed)
    coords = np.asarray(coords, float)
    D = cdist(coords, coords)
    K = matern_covariance(D, sigma2=float(sigma2), rho=float(rho), nu=float(nu))
    K[np.diag_indices_from(K)] += nugget
    L = cholesky(K)

    Z = rng.standard_normal((K.shape[0], n_realizations))
    F = L @ Z
    return F.T  # (n_realizations, n_space)


def reshape_predictor(X):
    X = np.asarray(X)
    return X.reshape(X.shape[0], -1, order="F") if X.ndim != 2 else X


def simulate_iid_temporal_gaussian_cholesky(
    nt: int,
    P: int,
    *,
    a: float = 0.01,
    alpha: float = 0.5,
    tau: float = 1.2,
    sigma: float = 1.0,
    nugget0: float = 1e-10,
    max_retries: int = 6,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    h = np.arange(nt, dtype=np.float64)
    gamma0 = sigma**2 / (1.0 + a * np.power(h, 2.0 * alpha))**tau
    gamma0[0] = sigma**2
    K = toeplitz(gamma0)

    nugget = nugget0
    for _ in range(max_retries + 1):
        try:
            L = cholesky(K + nugget * np.eye(nt))
            break
        except np.linalg.LinAlgError:
            nugget *= 10.0
    else:
        raise RuntimeError("Cholesky failed; try larger nugget.")

    Z = rng.standard_normal(size=(nt, P))
    return L @ Z


def ensemble_band(arr, q=(0.05, 0.5, 0.95), axis=0):
    return np.quantile(arr, q, axis=axis)


def temporal_acf_mean(Z, max_lag=50):
    # Z: (T, S)
    Z = np.asarray(Z, float)
    Zc = Z - np.nanmean(Z, axis=0, keepdims=True)
    var = np.nanmean(Zc**2, axis=0)
    out = np.zeros(max_lag + 1, dtype=float)
    out[0] = 1.0
    for h in range(1, max_lag + 1):
        c = np.nanmean(Zc[:-h] * Zc[h:], axis=0)
        out[h] = np.nanmean(c / (var + 1e-12))
    return out


def spatial_corr_binned(Z_t, coords, n_bins=20):
    Z_t = np.asarray(Z_t, float)
    coords = np.asarray(coords, float)
    S = Z_t.shape[0]
    D = cdist(coords, coords)
    tri = np.triu_indices(S, k=1)
    d = D[tri]

    zc = Z_t - np.nanmean(Z_t)
    prod = (zc[:, None] * zc[None, :])[tri]
    var = np.nanmean(zc**2)
    corr = prod / (var + 1e-12)

    edges = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    bin_id = np.digitize(d, edges) - 1

    h_cent, c_mean = [], []
    for b in range(n_bins):
        m = bin_id == b
        if not np.any(m):
            continue
        h_cent.append(np.mean(d[m]))
        c_mean.append(np.mean(corr[m]))
    return np.asarray(h_cent), np.asarray(c_mean)


# ============================================================
# Main synthetic study
# ============================================================
def main(cfg: SyntheticStudyConfig) -> None:
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir)
    logger.info("Starting synthetic study.")
    logger.info("Output dir: %s", out_dir)

    # Save config
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # -----------------------------
    # Spatial grid
    # -----------------------------
    s_coords = np.array(
        [(x, y) for x in np.linspace(0, 1, cfg.nsx) for y in np.linspace(0, 1, cfg.nsy)],
        dtype=float,
    )
    n = s_coords.shape[0]
    logger.info("Spatial grid: %dx%d => n_sites=%d", cfg.nsx, cfg.nsy, n)

    # -----------------------------
    # Time
    # -----------------------------
    nt = int(cfg.nt)
    n_test = int(cfg.n_test)
    t_coords = np.arange(nt, dtype=float)

    dates_all = pd.date_range(cfg.start_date, periods=nt, freq=cfg.freq)
    cutoff = nt - n_test
    train_mask = slice(0, cutoff)
    test_mask = slice(cutoff, None)
    dates_tr, dates_te = dates_all[train_mask], dates_all[test_mask]

    logger.info("Time: nt=%d, n_test=%d, train=%d", nt, n_test, cutoff)

    # -----------------------------
    # Covariates X1, X2 (temporal corr)
    # -----------------------------
    P = int(cfg.P)
    a1, alpha1, tau1, sigma1 = cfg.X1_kernel
    a2, alpha2, tau2, sigma2 = cfg.X2_kernel

    X1 = simulate_iid_temporal_gaussian_cholesky(
        nt, P, a=a1, alpha=alpha1, tau=tau1, sigma=sigma1, seed=cfg.seed_X1
    )
    X2 = simulate_iid_temporal_gaussian_cholesky(
        nt, P, a=a2, alpha=alpha2, tau=tau2, sigma=sigma2, seed=cfg.seed_X2
    )
    logger.info("Simulated covariates: X1, X2 shape=(%d,%d)", nt, P)

    # -----------------------------
    # Spatial coefficient fields b1(s), g1(s)
    # -----------------------------
    b1 = simulate_matern_coeff_matrix(
        s_coords, P, sigma2=cfg.b1_sigma2, rho=cfg.b1_rho, nu=cfg.b1_nu, seed=cfg.b1_seed
    )  # (P, n)
    g1 = simulate_matern_coeff_matrix(
        s_coords, P, sigma2=cfg.g1_sigma2, rho=cfg.g1_rho, nu=cfg.g1_nu, seed=cfg.g1_seed
    )  # (P, n)
    logger.info("Simulated coefficient fields: b1, g1 shape=(%d,%d)", P, n)

    # -----------------------------
    # mu(t,s), sigma(t,s)
    # -----------------------------
    beta0, gamma0 = float(cfg.beta0), float(cfg.gamma0)
    mu = beta0 + np.tanh(X1 @ b1)           # (nt, n)
    sig = np.exp(gamma0 + np.sin(X1 @ g1))  # (nt, n)
    logger.info("Built mu/sig: mu shape=%s sig shape=%s", mu.shape, sig.shape)

    # -----------------------------
    # Latent Z(t,s) and observed Y(t,s)
    # -----------------------------
    pars_Z = list(cfg.pars_Z)
    logger.info("Simulating latent Z with true params: %s", pars_Z)
    Z_full = simulate_gneiting_jax(
        spatial_coordinates=s_coords,
        temporal_coordinates=t_coords,
        params=pars_Z,
        L=int(cfg.L),
        chunk_size=int(cfg.chunk_size),
        nugget=float(cfg.gneiting_epsilon),
    )
    Z_full = np.asarray(Z_full, dtype=np.float32)
    Y_true = np.exp(mu + sig * Z_full).astype(np.float32)
    logger.info("Generated dataset: Z_full shape=%s, Y_true shape=%s", Z_full.shape, Y_true.shape)

    # Train/test
    X_all = reshape_predictor(X1).astype(np.float32)  # EASY scenario: use X1 only (like your code)
    X_tr, Y_tr = X_all[train_mask], Y_true[train_mask]
    X_te = X_all[test_mask]
    mu_te, sig_te = mu[test_mask], sig[test_mask]
    Z_te_true = Z_full[test_mask]
    logger.info("Split: X_tr=%s Y_tr=%s X_te=%s", X_tr.shape, Y_tr.shape, X_te.shape)

    # -----------------------------
    # Oracle ensemble on test window
    # -----------------------------
    logger.info("Simulating oracle ensemble on test window: n_sims_oracle=%d", cfg.n_sims_oracle)
    t_coords_test = np.arange(n_test, dtype=float)

    Y_oracle_sims = np.empty((cfg.n_sims_oracle, n_test, n), dtype=np.float32)
    for r in range(cfg.n_sims_oracle):
        Z_test_r = simulate_gneiting_jax(
            spatial_coordinates=s_coords,
            temporal_coordinates=t_coords_test,
            params=pars_Z,
            L=int(cfg.L),
            chunk_size=int(cfg.chunk_size),
            nugget=float(cfg.gneiting_epsilon),
        )
        Z_test_r = np.asarray(Z_test_r, dtype=np.float32)
        Y_oracle_sims[r] = np.exp(mu_te + sig_te * Z_test_r).astype(np.float32)
    # Put realized truth at index 0 (as in your experiment)
    Y_oracle_sims[0] = Y_true[test_mask]
    logger.info("Oracle ensemble ready: %s", Y_oracle_sims.shape)

    # -----------------------------
    # Fit methods & simulate
    # -----------------------------
    taus = np.linspace(0.01, 0.99, cfg.taus_n).astype(np.float32)
    nu_fixed_val = float(cfg.pars_Z[2])

    fitted: Dict[str, Any] = {}
    sims_Y: Dict[str, np.ndarray] = {}
    sims_Z: Dict[str, np.ndarray] = {}

    for method, marg_kwargs in cfg.methods:
        logger.info("Fitting method=%s ...", method)

        model = mlq4sModel(
            coords=s_coords,
            marginal_method=method,
            marginal_kwargs=dict(marg_kwargs),
            marginal_taus=taus,
            var_select=False,

            # Gneiting fit settings (close to your experiment)
            gneiting_strategy="balanced",
            gneiting_strata_bins=(8, 5),
            gneiting_initial_params=None,
            gneiting_estimate_nu=False,
            gneiting_nu_fixed=nu_fixed_val,
            block_size=int(cfg.gneiting_block_size),
            n_blocks=int(cfg.gneiting_n_blocks),
            t_max=int(cfg.gneiting_t_max),
            epsilon=float(cfg.gneiting_epsilon),
            random_state=int(cfg.random_state),
        ).fit(X_cov=X_tr, Y_obs=Y_tr, dates=dates_tr)

        fitted[method] = model
        logger.info("Selected marginal hyperparams (%s): %s", method, model.get_selected_hyperparams())
        logger.info("Estimated Gneiting params (%s): %s", method, getattr(model, "gneiting_params_", None))

        logger.info("Simulating method=%s: n_sims=%d", method, cfg.n_sims_per_method)
        _, Z_sim, Y_sim = model.simulate(
            X_test=X_te,
            test_dates=dates_te,
            n_simulations=int(cfg.n_sims_per_method),
            L_draws=int(cfg.L),
            chunk_size=int(cfg.chunk_size),
            nugget=float(cfg.gneiting_epsilon),
        )

        sims_Z[method] = np.asarray(Z_sim, dtype=np.float32)
        sims_Y[method] = np.asarray(Y_sim, dtype=np.float32)

    # -----------------------------
    # Save arrays (compact)
    # -----------------------------
    np.savez_compressed(
        out_dir / "synthetic_study_outputs.npz",
        coords=s_coords.astype(np.float32),
        dates_tr=np.asarray(dates_tr.astype(str)),
        dates_te=np.asarray(dates_te.astype(str)),
        X_tr=X_tr,
        Y_tr=Y_tr,
        X_te=X_te,
        Y_te_truth=Y_true[test_mask],
        Z_te_truth=Z_te_true,
        mu_te=mu_te.astype(np.float32),
        sig_te=sig_te.astype(np.float32),
        Y_oracle_sims=Y_oracle_sims,
        **{f"Ysim_{k}": v for k, v in sims_Y.items()},
        **{f"Zsim_{k}": v for k, v in sims_Z.items()},
    )
    logger.info("Saved arrays to %s", out_dir / "synthetic_study_outputs.npz")

    # ============================================================
    # Diagnostics & figures
    # ============================================================

    # ---- 1) Time-series bands at a few sites
    logger.info("Plotting time-series bands...")
    q_or = ensemble_band(Y_oracle_sims, q=(0.05, 0.5, 0.95), axis=0)  # (3, n_test, n)
    t = np.arange(n_test)

    for s in cfg.sites_to_plot:
        s = int(s)
        if s < 0 or s >= n:
            continue

        plt.figure(figsize=(10, 3))
        plt.fill_between(t, q_or[0, :, s], q_or[2, :, s], alpha=0.20, label="Oracle 5–95%")
        plt.plot(t, q_or[1, :, s], lw=1.5, label="Oracle median")
        plt.plot(t, Y_true[test_mask][:, s], lw=1.0, alpha=0.85, label="Truth")

        for method, Y_sim in sims_Y.items():
            q_m = ensemble_band(Y_sim, q=(0.05, 0.5, 0.95), axis=0)
            plt.plot(t, q_m[1, :, s], lw=1.0, alpha=0.9, label=f"{method} median")

        plt.title(f"Test window — site {s}")
        plt.xlabel("Time index (test)")
        plt.ylabel("Y")
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / f"time_series_site_{s}.png", dpi=200)
        plt.close()

    # ---- 2) Pooled QQ vs oracle
    logger.info("Plotting pooled QQ...")
    oracle_pool = Y_oracle_sims.reshape(Y_oracle_sims.shape[0], -1)
    qs = np.linspace(0.01, 0.99, 99)

    # oracle quantile distribution across oracle ensemble
    oracle_q = np.quantile(oracle_pool, qs, axis=1)  # (len(qs), n_sims_oracle)
    oracle_q_low = np.quantile(oracle_q, 0.05, axis=1)
    oracle_q_med = np.quantile(oracle_q, 0.50, axis=1)
    oracle_q_hi = np.quantile(oracle_q, 0.95, axis=1)

    plt.figure(figsize=(5.6, 5.6))
    plt.fill_between(oracle_q_med, oracle_q_low, oracle_q_hi, alpha=0.20, label="Oracle quantile band (5–95%)")
    plt.plot(oracle_q_med, oracle_q_med, lw=1.0, alpha=0.6)

    for method, Y_sim in sims_Y.items():
        sim_pool = Y_sim.reshape(Y_sim.shape[0], -1)
        sim_q_med = np.quantile(sim_pool, qs, axis=1)  # median across sims per quantile level
        plt.plot(oracle_q_med, sim_q_med, lw=1.6, label=method)

    plt.xlabel("Oracle quantiles")
    plt.ylabel("Model quantiles (median across sims)")
    plt.title("Pooled QQ — test window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "pooled_qq.png", dpi=220)
    plt.close()

    # ---- 3) Temporal ACF in latent space
    logger.info("Plotting temporal ACF (latent Z)...")
    acf_true = temporal_acf_mean(Z_te_true, max_lag=cfg.max_lag_acf)

    plt.figure(figsize=(6.8, 3.6))
    plt.plot(acf_true, lw=2.0, label="Truth (latent Z)")

    for method, Z_sim in sims_Z.items():
        # average ACF over sims
        acfs = []
        for i in range(Z_sim.shape[0]):
            acfs.append(temporal_acf_mean(Z_sim[i], max_lag=cfg.max_lag_acf))
        acf_m = np.mean(np.stack(acfs, axis=0), axis=0)
        plt.plot(acf_m, lw=1.5, label=method)

    plt.title("Temporal ACF in latent space (test window)")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "latent_temporal_acf.png", dpi=220)
    plt.close()

    # ---- 4) Spatial correlation binned at first test day
    logger.info("Plotting spatial correlation (binned) at first test day...")
    t0 = 0
    h_true, c_true = spatial_corr_binned(Z_te_true[t0], s_coords, n_bins=cfg.spatial_bins)

    plt.figure(figsize=(6.8, 3.6))
    plt.plot(h_true, c_true, marker="o", ms=3, lw=1.5, label="Truth (latent Z)")

    for method, Z_sim in sims_Z.items():
        cs = []
        for i in range(Z_sim.shape[0]):
            h_m, c_m = spatial_corr_binned(Z_sim[i, t0], s_coords, n_bins=cfg.spatial_bins)
            cs.append(np.interp(h_true, h_m, c_m, left=np.nan, right=np.nan))
        c_avg = np.nanmean(np.stack(cs, axis=0), axis=0)
        plt.plot(h_true, c_avg, marker=".", ms=3, lw=1.5, label=method)

    plt.title("Spatial correlation (binned) — latent space (first test day)")
    plt.xlabel("Distance")
    plt.ylabel("Correlation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "latent_spatial_corr_binned.png", dpi=220)
    plt.close()

    logger.info("All figures saved under %s", fig_dir)
    logger.info("Synthetic study complete.")


if __name__ == "__main__":
    cfg = SyntheticStudyConfig()
    main(cfg)
