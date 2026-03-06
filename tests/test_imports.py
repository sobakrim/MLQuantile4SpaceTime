import pytest
import numpy as np

from mlq4st.st_grf import simulate_gneiting_jax


def test_simulate_gneiting_jax_runs_small():
    # tiny shapes so CI is fast
    coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)  # (n_s=2, d=2)
    t = np.arange(5, dtype=float)                              # (n_t=5,)
    params = [0.7, 3.4, 1.2, 0.3, 0.9, 0.7]                     # [a,kappa,nu,alpha,tau,q]

    Z = simulate_gneiting_jax(coords, t, params, L=100, chunk_size=50, nugget=1e-6)

    assert Z.shape == (5, 2)
