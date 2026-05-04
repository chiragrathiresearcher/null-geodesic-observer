# ============================================================
# NGO — Null Geodesic Observer
# Module : ngo/core/path_integral.py
# Author : Chirag Rathi
# Purpose: Compute L_A = ∫ n(x) dσ  (equation 3.5 in paper)
# ============================================================
"""
path_integral.py
----------------
Computes the optical path length along a null geodesic:

    L = ∫_γ n(x) dσ                        (equation 3.5)

where n(x) = √(−g₀₀) is the effective refractive index of
curved spacetime, and dσ = √(g_ij dx^i dx^j) is the spatial
line element.

In the weak-field limit:
    n(x) ≈ 1 + Φ(x)/c²                     (equation 3.6)

giving:
    L = d + (1/c²) ∫_γ Φ(x) dσ             (equation 3.7)
"""

import numpy as np
from .geodesic import GeodesicSolution
from .metric import Metric, WeakFieldMetric

C = 2.998e8   # speed of light [m/s]


def effective_refractive_index(metric: Metric, x: np.ndarray) -> float:
    """
    Compute the effective refractive index n(x) = √(−g₀₀(x)).

    In flat spacetime g₀₀ = −c², so n = c (normalised to 1 for c=1 units).
    In curved spacetime n(x) > 1 near masses → photons slow down.

    Parameters
    ----------
    metric : Metric
    x      : np.ndarray (4,) — coordinates

    Returns
    -------
    float — n(x)
    """
    g = metric.g(x)
    g00 = g[0, 0]
    if g00 >= 0:
        raise ValueError(
            f"g₀₀ = {g00:.4f} ≥ 0 at x={x}. "
            "Metric has wrong signature or coordinates are invalid."
        )
    return float(np.sqrt(-g00))


def spatial_line_element(
    metric: Metric,
    x: np.ndarray,
    dx_spatial: np.ndarray
) -> float:
    """
    Compute the spatial line element dσ = √(g_ij dx^i dx^j).

    Parameters
    ----------
    metric     : Metric
    x          : np.ndarray (4,) — current position
    dx_spatial : np.ndarray (3,) — spatial displacement [dx¹, dx², dx³]

    Returns
    -------
    float — dσ
    """
    g = metric.g(x)
    g_spatial = g[1:, 1:]   # 3×3 spatial block
    ds2 = np.einsum('ij,i,j', g_spatial, dx_spatial, dx_spatial)
    if ds2 < 0:
        ds2 = 0.0   # numerical noise
    return float(np.sqrt(ds2))


def compute_path_integral(sol: GeodesicSolution) -> dict:
    """
    Compute the optical path length L = ∫ n(x) dσ along a geodesic.

    Uses trapezoidal integration over the discrete geodesic points.

    Parameters
    ----------
    sol : GeodesicSolution — output from GeodesicSolver.solve()

    Returns
    -------
    dict with keys:
        'L'          : float — total optical path length  [m]
        't_travel'   : float — coordinate travel time = L/c  [s]
        'n_values'   : np.ndarray — refractive index n(x) at each step
        'dsigma'     : np.ndarray — spatial step sizes dσ
        'integrand'  : np.ndarray — n(x) × dσ at each step
        'n_steps'    : int   — number of integration steps used
    """
    x_path = sol.x          # shape (4, N)
    N      = x_path.shape[1]
    metric = sol.metric

    n_values  = np.zeros(N)
    dsigma    = np.zeros(N - 1)
    integrand = np.zeros(N - 1)

    # compute refractive index at each point
    for i in range(N):
        try:
            n_values[i] = effective_refractive_index(metric, x_path[:, i])
        except Exception:
            n_values[i] = 1.0   # fallback to flat

    # compute spatial steps and integrand
    for i in range(N - 1):
        dx_sp = x_path[1:, i+1] - x_path[1:, i]
        dsigma[i]    = spatial_line_element(metric, x_path[:, i], dx_sp)
        n_mid        = 0.5 * (n_values[i] + n_values[i+1])
        integrand[i] = n_mid * dsigma[i]

    L = float(np.sum(integrand))
    t_travel = L / C

    return {
        'L'         : L,
        't_travel'  : t_travel,
        'n_values'  : n_values,
        'dsigma'    : dsigma,
        'integrand' : integrand,
        'n_steps'   : N
    }


def compute_weak_field_integral(
    sol       : GeodesicSolution,
    metric    : WeakFieldMetric
) -> dict:
    """
    Compute the weak-field path integral explicitly:

        L = d + (1/c²) ∫_γ Φ(x) dσ            (equation 3.7)
        Δt = (1/c³) [∫_γ₁ Φdσ₁ − ∫_γ₂ Φdσ₂]  (equation 3.8)

    This gives direct access to the potential integral along
    the geodesic, useful for comparing two paths.

    Parameters
    ----------
    sol    : GeodesicSolution
    metric : WeakFieldMetric — needed for potential evaluation

    Returns
    -------
    dict with keys:
        'potential_integral' : float — ∫ Φ(x) dσ  [J·m/kg]
        'd_coord'            : float — coordinate path length  [m]
        'L_corrected'        : float — d + (1/c²)∫Φdσ  [m]
        't_travel'           : float — travel time  [s]
        'phi_values'         : np.ndarray — Φ(x) at each step
    """
    x_path = sol.x
    N      = x_path.shape[1]

    phi_values = np.zeros(N)
    dsigma     = np.zeros(N - 1)
    phi_integrand = np.zeros(N - 1)
    d_steps    = np.zeros(N - 1)

    for i in range(N):
        phi_values[i] = metric.potential(x_path[:, i])

    for i in range(N - 1):
        dx_sp = x_path[1:, i+1] - x_path[1:, i]
        dsigma[i]  = spatial_line_element(metric, x_path[:, i], dx_sp)
        d_steps[i] = dsigma[i]
        phi_mid    = 0.5 * (phi_values[i] + phi_values[i+1])
        phi_integrand[i] = phi_mid * dsigma[i]

    potential_integral = float(np.sum(phi_integrand))
    d_coord            = float(np.sum(d_steps))
    L_corrected        = d_coord + potential_integral / C**2
    t_travel           = L_corrected / C

    return {
        'potential_integral': potential_integral,
        'd_coord'           : d_coord,
        'L_corrected'       : L_corrected,
        't_travel'          : t_travel,
        'phi_values'        : phi_values,
    }
