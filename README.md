# MLQuantile4SpaceTime
Machine-learning quantile regression for space-time processes

[![HAL](https://img.shields.io/badge/HAL-hal--05441043-B03532)](https://hal.science/hal-05441043/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MLQuantile4SpaceTime** is a Python package for **conditional distribution modeling** and **simulation of spatio-temporal processes**. It combines **machine-learning quantile regression** (to learn non-Gaussian, covariate-dependent marginals) with a **latent Gaussian random field (GRF)** (to enforce coherent space-time dependence).

---

## Paper

- HAL preprint: https://hal.science/hal-05441043/

---

## Features

- **Conditional marginals** `Y | X` via quantile regression:
  - KNN-based conditional CDF (`knn`)
  - Quantile Regression Forests (`qrf`)
  - Quantile Regression Neural Networks (`qrnn`, via `quantnn`)
- **Latent Gaussian mapping** (Gaussian copula):
  - `U = F_{Y|X}(y)`
  - `Z = Phi^{-1}(U)`  (Phi = standard normal CDF)
- **Spatio-temporal dependence** in latent space with GRFs (e.g., Matérn–Gneiting)
- End-to-end **fit → simulate → invert** workflow:
  - fit marginals → map `Y -> Z` → fit GRF → simulate `Z` → map `Z -> Y`
- Optional hyperparameter selection via time-series cross-validation (depending on method)

---

## Installation

### From GitHub (requires `git`)
```bash
pip install "git+https://github.com/sobakrim/MLQuantile4SpaceTime.git@main"
