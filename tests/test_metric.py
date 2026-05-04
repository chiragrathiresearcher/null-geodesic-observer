# ============================================================
# NGO — Null Geodesic Observer
# Tests  : tests/test_metric.py
# Author : Chirag Rathi
# ============================================================
"""
Unit tests for ngo.core.metric

Tests:
  - Minkowski metric is diagonal [-1,1,1,1]
  - Schwarzschild metric recovers Minkowski at r → ∞
  - Christoffel symbols are zero for Minkowski
  - Weak field metric has correct potential sign
  - Null check returns 0 for valid null vectors
  - CustomMetric raises on wrong shape
"""

import numpy as np
import pytest
from ngo.core.metric import (
    MinkowskiMetric, SchwarzschildMetric,
    WeakFieldMetric, CustomMetric, get_metric
)

C   = 2.998e8
G   = 6.674e-11
M_S = 1.989e30


# ── Minkowski ──────────────────────────────────────────────
class TestMinkowskiMetric:

    def setup_method(self):
        self.m = MinkowskiMetric()
        self.x = np.array([0.0, 1.0, 0.0, 0.0])

    def test_signature(self):
        g = self.m.g(self.x)
        assert g[0, 0] == pytest.approx(-1.0)
        assert g[1, 1] == pytest.approx(1.0)
        assert g[2, 2] == pytest.approx(1.0)
        assert g[3, 3] == pytest.approx(1.0)

    def test_diagonal(self):
        g = self.m.g(self.x)
        off_diag = g - np.diag(np.diag(g))
        assert np.allclose(off_diag, 0.0)

    def test_christoffel_zero(self):
        Gamma = self.m.christoffel(self.x)
        assert np.allclose(Gamma, 0.0)

    def test_null_check(self):
        # null vector in x-direction: k = (1, 1, 0, 0) → g_μν k^μ k^ν = -1+1 = 0
        k = np.array([1.0, 1.0, 0.0, 0.0])
        val = self.m.null_check(self.x, k)
        assert val == pytest.approx(0.0, abs=1e-12)

    def test_inverse(self):
        g     = self.m.g(self.x)
        g_inv = self.m.g_inv(self.x)
        product = g @ g_inv
        assert np.allclose(product, np.eye(4), atol=1e-10)


# ── Schwarzschild ──────────────────────────────────────────
class TestSchwarzschildMetric:

    def setup_method(self):
        self.m  = SchwarzschildMetric(M=M_S)
        self.rs = 2 * G * M_S / C**2

    def test_schwarzschild_radius(self):
        assert self.m.rs == pytest.approx(self.rs, rel=1e-6)

    def test_far_field_limit(self):
        """At r >> r_s, should approximate Minkowski."""
        x = np.array([0.0, 1e13, np.pi/2, 0.0])   # 1e13 m >> r_s ≈ 2953 m
        g = self.m.g(x)
        assert g[0, 0] == pytest.approx(-C**2, rel=1e-4)
        assert g[1, 1] == pytest.approx(1.0,   rel=1e-4)

    def test_raises_inside_horizon(self):
        x_inside = np.array([0.0, self.rs * 0.5, np.pi/2, 0.0])
        with pytest.raises(ValueError):
            self.m.g(x_inside)

    def test_metric_symmetric(self):
        x = np.array([0.0, 1e10, np.pi/2, 0.0])
        g = self.m.g(x)
        assert np.allclose(g, g.T)

    def test_determinant_negative(self):
        """Lorentzian signature: det(g) < 0."""
        x = np.array([0.0, 1e10, np.pi/2, 0.0])
        g = self.m.g(x)
        assert np.linalg.det(g) < 0


# ── WeakFieldMetric ────────────────────────────────────────
class TestWeakFieldMetric:

    def setup_method(self):
        self.masses = [(M_S, np.array([0., 0., 0.]))]
        self.m = WeakFieldMetric(self.masses)

    def test_potential_negative(self):
        """Gravitational potential is negative (bound state)."""
        x = np.array([0., 1e11, 0., 0.])
        phi = self.m.potential(x)
        assert phi < 0.0

    def test_potential_decreases_with_distance(self):
        x1 = np.array([0., 1e10, 0., 0.])
        x2 = np.array([0., 1e11, 0., 0.])
        phi1 = self.m.potential(x1)
        phi2 = self.m.potential(x2)
        # |phi1| > |phi2| (closer = deeper potential)
        assert abs(phi1) > abs(phi2)

    def test_flat_at_infinity(self):
        """Very far from mass, metric approaches Minkowski."""
        x = np.array([0., 1e20, 0., 0.])
        g = self.m.g(x)
        assert g[0, 0] == pytest.approx(-1.0, rel=1e-6)
        assert g[1, 1] == pytest.approx(1.0,  rel=1e-6)

    def test_g00_more_negative_near_mass(self):
        """g₀₀ = -(1 + 2Φ/c²): near mass Φ<0 so |g₀₀| < 1."""
        x_near = np.array([0., 1e9,  0., 0.])
        x_far  = np.array([0., 1e15, 0., 0.])
        g_near = self.m.g(x_near)
        g_far  = self.m.g(x_far)
        assert abs(g_near[0, 0]) < abs(g_far[0, 0])


# ── CustomMetric ───────────────────────────────────────────
class TestCustomMetric:

    def test_valid_custom(self):
        def flat(x):
            return np.diag([-1., 1., 1., 1.])
        m = CustomMetric(flat)
        g = m.g(np.zeros(4))
        assert g.shape == (4, 4)

    def test_wrong_shape_raises(self):
        def bad(x):
            return np.eye(3)
        m = CustomMetric(bad)
        with pytest.raises(ValueError):
            m.g(np.zeros(4))

    def test_not_callable_raises(self):
        with pytest.raises(TypeError):
            CustomMetric("not_a_function")


# ── get_metric factory ─────────────────────────────────────
class TestGetMetric:

    def test_minkowski(self):
        m = get_metric('minkowski')
        assert isinstance(m, MinkowskiMetric)

    def test_schwarzschild(self):
        m = get_metric('schwarzschild', M=M_S)
        assert isinstance(m, SchwarzschildMetric)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_metric('kerr')   # not yet implemented
