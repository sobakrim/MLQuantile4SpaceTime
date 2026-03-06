# tests/test_optional_jax.py
import pytest
from MLQuantile4SpaceTime.st_grf import simulate_gneiting_jax

def test_simulate_requires_jax():
    with pytest.raises(ImportError):
        simulate_gneiting_jax(None, None, None, L=1)
