# MLQuantile4SpaceTime : Machine Learning Quantile regression and latent Gaussian random fields for Space-Time processes

[![HAL](https://img.shields.io/badge/HAL-hal--05441043-B03532)](https://hal.science/hal-05441043/)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


A research codebase for **modeling and simulating spatio-temporal processes** by combining  
**machine-learning quantile regression** with **latent Gaussian random fields (GRFs)**.

---

## Paper

**HAL preprint**: https://hal.science/hal-05441043/

---

## Overview

**Quantile2SpaceTime** implements a coherent two-stage framework:

1. **Conditional marginals (data space)**  
   Learn the conditional distribution \(Y \mid X\) using quantile regression:
   - **KNN CDF**
   - **Quantile Regression Forests (QRF)**
   - **Quantile Regression Neural Networks (QRNN)**

2. **Dependence (latent space)**  
   Map observations to a latent Gaussian space and model dependence with a GRF:
   - Transform: \(U = F_{Y|X}(y)\), then \(Z = \Phi^{-1}(U)\)
   - Fit a spatio-temporal GRF for \(Z(s,t)\) (e.g., **Matérn–Gneiting**)
   - Simulate \(Z\) and invert back to \(Y\)

This yields simulations that preserve:
- **non-Gaussian, covariate-dependent marginals**, and
- **spatio-temporal dependence** through the latent GRF.
