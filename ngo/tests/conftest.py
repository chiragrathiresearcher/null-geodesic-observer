# ============================================================
# NGO — Null Geodesic Observer
# File   : tests/conftest.py
# Author : Chirag Rathi
# Purpose: Shared pytest fixtures
# ============================================================

import numpy as np
import pytest
from ngo.core.metric import (
    MinkowskiMetric, SchwarzschildMetric, WeakFieldMetric
)

C   = 2.998e8
G   = 6.674e-11
M_S = 1.989e30


@pytest.fixture
def minkowski():
    return MinkowskiMetric()


@pytest.fixture
def schwarzschild_solar():
    return SchwarzschildMetric(M=M_S)


@pytest.fixture
def weak_field_solar():
    """Single solar mass at origin."""
    return WeakFieldMetric(masses=[(M_S, np.array([0., 0., 0.]))])


@pytest.fixture
def far_position():
    """Position far from any mass — effectively flat spacetime."""
    return np.array([0., 1e15, 0., 0.])


@pytest.fixture
def near_position():
    """Position close to Sun — curved spacetime regime."""
    return np.array([0., 1e9, 0., 0.])


@pytest.fixture
def null_vector_x():
    """Null 4-vector propagating in x-direction (Minkowski)."""
    return np.array([1., 1., 0., 0.])
