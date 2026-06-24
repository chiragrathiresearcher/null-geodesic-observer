# Tests for Schwarzschild coordinate sanity checks in DeltaTComputer
import numpy as np
import pytest

from ngo.core.metric import SchwarzschildMetric
from ngo.core.delta_t import DeltaTComputer


def test_schwarzschild_coord_check_raises_on_r_inside_horizon():
    m = SchwarzschildMetric(M=1.989e30)
    comp = DeltaTComputer(m)

    # place the source inside the Schwarzschild radius to trigger the check
    x_bad = np.array([0.0, 0.5 * m.rs, np.pi / 2, 0.0])
    k_bad = np.array([1.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match=r"r_s"):
        comp.compute(x_bad, k_bad, x_bad, k_bad, lam_end=1.0, n_eval=10)


def test_schwarzschild_coord_check_raises_on_bad_shape():
    m = SchwarzschildMetric(M=1.989e30)
    comp = DeltaTComputer(m)

    # invalid coordinate array (too short) should raise a clear error
    x_bad_shape = np.array([0.0, 1.0])
    k0 = np.array([1.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match=r"length-4 array"):
        comp.compute(x_bad_shape, k0, x_bad_shape, k0, lam_end=1.0, n_eval=10)
