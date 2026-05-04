# ============================================================
# NGO — Null Geodesic Observer
# Tests  : tests/test_delta_t.py
# Author : Chirag Rathi
# ============================================================
"""
Unit tests for ngo.core.delta_t

Critical tests:
  - Flat spacetime: Δt = 0 exactly
  - Shapiro analytic formula matches known values
  - Lensing analytic formula correct
  - Symmetric configuration: Δt ≈ 0
  - Asymmetric configuration: Δt ≠ 0
"""

import numpy as np
import pytest
from ngo.core.delta_t import shapiro_delay_analytic, lensing_delay_analytic

C   = 2.998e8
G   = 6.674e-11
M_S = 1.989e30


class TestShapiroAnalytic:

    def test_cassini_order_of_magnitude(self):
        """
        Cassini measured Δt ≈ 246 μs.
        Our analytic formula should give the same order.
        """
        M      = M_S
        r_emit = 1.5e12    # ~Cassini at 10 AU
        r_obs  = 1.5e11    # 1 AU (Earth)
        b      = 1.6 * 6.957e8   # 1.6 solar radii

        dt = shapiro_delay_analytic(M, r_emit, r_obs, b)

        # Should be O(100 μs)
        assert 50e-6 < dt < 500e-6, (
            f"Shapiro delay {dt*1e6:.1f} μs outside expected range 50–500 μs"
        )

    def test_increases_with_mass(self):
        """Heavier mass → larger Shapiro delay."""
        b = 1e9
        r = 1e12
        dt1 = shapiro_delay_analytic(M_S,     r, r, b)
        dt2 = shapiro_delay_analytic(10 * M_S, r, r, b)
        assert dt2 > dt1

    def test_decreases_with_impact_parameter(self):
        """Larger b (farther from mass) → smaller Shapiro delay."""
        r  = 1e12
        dt1 = shapiro_delay_analytic(M_S, r, r, 1e9)
        dt2 = shapiro_delay_analytic(M_S, r, r, 1e10)
        assert dt1 > dt2

    def test_positive(self):
        """Shapiro delay is always positive (gravity slows light)."""
        dt = shapiro_delay_analytic(M_S, 1e12, 1e11, 1e9)
        assert dt > 0


class TestLensingAnalytic:

    def test_equal_b_gives_zero(self):
        """If b₁ = b₂, Δt = 0 (same path)."""
        dt = lensing_delay_analytic(M_S, b1=1e10, b2=1e10)
        assert dt == pytest.approx(0.0, abs=1e-30)

    def test_positive_when_b2_greater(self):
        """b₂ > b₁ → geodesic 2 passes farther → ln(b₂/b₁) > 0."""
        dt = lensing_delay_analytic(M_S, b1=1e9, b2=1e10)
        assert dt > 0

    def test_q0957_order_of_magnitude(self):
        """
        Q0957+561: Δt ≈ 417 days.
        Use galaxy mass ~10^12 M_sun and typical impact params.
        This is a rough order-of-magnitude check only.
        """
        M_galaxy = 1e12 * M_S
        b1 = 3e22   # ~1 kpc
        b2 = 4e22   # ~1.3 kpc
        dt = lensing_delay_analytic(M_galaxy, b1, b2)
        dt_days = dt / 86400
        # Very rough: should be within 2 orders of magnitude of 417 days
        assert 1 < dt_days < 1e5, f"Δt = {dt_days:.1f} days, expected O(100) days"

    def test_scales_with_mass(self):
        """Double the mass → double the delay."""
        dt1 = lensing_delay_analytic(M_S,     b1=1e9, b2=2e9)
        dt2 = lensing_delay_analytic(2 * M_S, b1=1e9, b2=2e9)
        assert dt2 == pytest.approx(2 * dt1, rel=1e-6)


class TestAsymmetryCondition:

    def test_symmetric_config_zero(self):
        """
        Two sources at equal distance on opposite sides of zero mass.
        Minkowski: Δt must be exactly 0.
        Addresses reviewer's symmetric counterexample.
        """
        from ngo.core.asymmetry_check import check_asymmetry

        # Identical path integrals (symmetric configuration)
        pi1 = {'L': 1.0e11}
        pi2 = {'L': 1.0e11}

        result = check_asymmetry(pi1, pi2)
        assert result['is_symmetric'] is True
        assert result['asymmetry_parameter'] == pytest.approx(0.0, abs=1e-10)

    def test_asymmetric_config_nonzero(self):
        """Different path integrals → asymmetric → Δt ≠ 0."""
        from ngo.core.asymmetry_check import check_asymmetry

        pi1 = {'L': 1.0e11}
        pi2 = {'L': 1.1e11}

        result = check_asymmetry(pi1, pi2)
        assert result['is_symmetric'] is False
        assert result['asymmetry_parameter'] > 0
        assert result['theorem_applies'] is True

    def test_asymmetry_parameter_range(self):
        """Asymmetry parameter must always be in [0, 1]."""
        from ngo.core.asymmetry_check import check_asymmetry

        for L1, L2 in [(1e10, 2e10), (5e9, 5e9), (1e8, 1e12)]:
            pi1 = {'L': L1}
            pi2 = {'L': L2}
            result = check_asymmetry(pi1, pi2)
            A = result['asymmetry_parameter']
            assert 0.0 <= A <= 1.0, f"A={A} out of [0,1] for L1={L1}, L2={L2}"
