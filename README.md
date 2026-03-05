# Quantile2SpaceTime

A general framework for **modeling and simulating spatio-temporal processes** by combining **machine-learning quantile regression** with **latent Gaussian random fields (GRFs)**. This repository accompanies the HAL preprint *“Combining machine learning quantile regression and Gaussian random fields: a general framework for modeling and simulating space-time processes”* by Obakrim, Allard, Benoit & Mariethoz. :contentReference[oaicite:0]{index=0}

---

## What this repo provides

**Quantile2SpaceTime** implements a two-stage (but fully coherent) pipeline:

1. **Learn conditional marginals** \( Y \mid X \) using quantile regression  
   - Methods: **KNN CDF**, **QRF**, **QRNN**
2. Map observations to a **latent Gaussian space**  
   - \( U = F_{Y|X}(y) \), then \( Z = \Phi^{-1}(U) \)
3. Model \( Z(s,t) \) with a **spatio-temporal GRF** (e.g., Matérn–Gneiting)
4. **Simulate** coherent spatio-temporal trajectories in latent space and **invert** back to data space \( Y \)

This yields simulations that respect:
- the **site-wise conditional distributions** learned from data (non-Gaussian, heteroskedastic, covariate-driven),
- and the **spatio-temporal dependence** enforced by the latent GRF.

---

## Repository highlights

- `SitewiseMarginal`:
  - KNN / QRF / QRNN
  - optional variable selection
  - time-series cross-validation with pinball loss
- `GneitingModel`:
  - composite likelihood estimation for Matérn–Gneiting space-time covariance
  - balanced / anchor / random block strategies
  - option to estimate or fix the Matérn smoothness `nu`
- `simulate_gneiting_jax`:
  - JAX-based scalable simulation with chunking

---

## Installation

### Option 1 — editable install (recommended for dev)
```bash
pip install -e .
