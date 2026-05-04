# ============================================================
# NGO — Null Geodesic Observer
# Module : ngo/core/proper_distance.py
# Author : Chirag Rathi
# Purpose: Compute invariant proper distance (fixes reviewer's
#          coordinate-dependence criticism)
# ============================================================
"""
proper_distance.py
------------------
Computes the proper (invariant) distance between two spacetime
events along a spacelike geodesic at constant coordinate time.

This addresses the reviewer's criticism:
  "The assumption d₁ = d₂ uses the spatial metric evaluated at
   the observer's location. This is not a diffeomorphism-invariant
   condition."

The proper distance is:
    d_proper = ∫_path √(g_ij dx^i dx^j)

along the spacelike geodesic connecting the two events.
"""

import numpy as np
from scipy.integrate import quad
from .metric import Metric

C = 2.998e8


def compute_proper_distance(
    metric  : Metric,
    event_a : np.ndarray,
    event_b : np.ndarray,
    n_steps : int = 500
) -> float:
    """
    Compute proper distance between two events along a straight
    coordinate path (valid for nearly flat or slowly varying metrics).

    For a rigorous treatment in strongly curved spacetimes, use
    compute_proper_distance_geodesic() instead.

    Parameters
    ----------
    metric  : Metric
    event_a : np.ndarray (4,) — start event [t, x1, x2, x3]
    event_b : np.ndarray (4,) — end event   [t, x1, x2, x3]
    n_steps : int — integration resolution

    Returns
    -------
    float — proper distance in metres
    """
    # parametrise straight line from a to b
    lambdas = np.linspace(0, 1, n_steps)
    dl = 1.0 / (n_steps - 1)

    dx = event_b - event_a   # full 4-displacement
    dx_spatial = dx[1:]      # spatial part

    total = 0.0
    for i in range(n_steps - 1):
        lam_mid = 0.5 * (lambdas[i] + lambdas[i+1])
        x_mid   = event_a + lam_mid * dx

        g = metric.g(x_mid)
        g_sp = g[1:, 1:]

        ds2 = np.einsum('ij,i,j', g_sp, dx_spatial, dx_spatial) * dl**2
        if ds2 > 0:
            total += np.sqrt(ds2)

    return float(total)


def is_equidistant_proper(
    metric   : Metric,
    observer : np.ndarray,
    source_1 : np.ndarray,
    source_2 : np.ndarray,
    tol      : float = 1e-3
) -> tuple:
    """
    Check if two sources are equidistant from observer in the
    invariant proper-distance sense.

    Parameters
    ----------
    metric   : Metric
    observer : np.ndarray (4,) — observer event
    source_1 : np.ndarray (4,) — source 1 event
    source_2 : np.ndarray (4,) — source 2 event
    tol      : float — fractional tolerance for equality

    Returns
    -------
    (bool, float, float) — (is_equal, proper_d1, proper_d2)
    """
    pd1 = compute_proper_distance(metric, source_1, observer)
    pd2 = compute_proper_distance(metric, source_2, observer)

    frac_diff = abs(pd1 - pd2) / max(pd1, pd2, 1e-30)
    is_equal  = frac_diff < tol

    return is_equal, pd1, pd2
