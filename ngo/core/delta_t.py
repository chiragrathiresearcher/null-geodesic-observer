# ============================================================
# NGO — Null Geodesic Observer
# Module : ngo/core/delta_t.py
# Author : Chirag Rathi
# Purpose: Compute Δt between two null geodesics (equation 3.10)
# ============================================================
"""
delta_t.py
----------
Central computation module. Computes the light travel time
difference Δt = t₁ − t₂ between two null geodesics from
equidistant sources to a common observer.

This is the core observable of the NGO framework, corresponding
to equation (3.10) in Rathi (2026):

    Δt = (1/c) Δ∫ [g_μν k^μ_⊥ k^ν_⊥]^½ dλ

Results include:
  - Δt in seconds
  - The asymmetry condition (is this a symmetric case?)
  - Comparison with known analytic formulae
  - Physical interpretation
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .geodesic import GeodesicSolution, GeodesicSolver
from .path_integral import compute_path_integral, compute_weak_field_integral
from .metric import Metric, WeakFieldMetric
from .proper_distance import compute_proper_distance
from .asymmetry_check import check_asymmetry

C = 2.998e8   # speed of light [m/s]
G = 6.674e-11
M_SUN = 1.989e30


@dataclass
class DeltaTResult:
    """
    Complete result of a Δt computation between two null geodesics.

    Attributes
    ----------
    delta_t          : float  — time difference t₁ − t₂  [seconds]
    t1               : float  — travel time of geodesic 1  [seconds]
    t2               : float  — travel time of geodesic 2  [seconds]
    L1               : float  — optical path length 1  [metres]
    L2               : float  — optical path length 2  [metres]
    d1               : float  — coordinate distance source 1  [metres]
    d2               : float  — coordinate distance source 2  [metres]
    proper_d1        : float  — proper distance source 1  [metres]
    proper_d2        : float  — proper distance source 2  [metres]
    is_symmetric     : bool   — True if Δt ≈ 0 by symmetry
    asymmetry_param  : float  — quantifies degree of asymmetry
    geodesic_1       : GeodesicSolution
    geodesic_2       : GeodesicSolution
    method           : str    — 'general' | 'weak_field'
    notes            : str    — physical interpretation
    """
    delta_t         : float
    t1              : float
    t2              : float
    L1              : float
    L2              : float
    d1              : float
    d2              : float
    proper_d1       : float
    proper_d2       : float
    is_symmetric    : bool
    asymmetry_param : float
    geodesic_1      : GeodesicSolution
    geodesic_2      : GeodesicSolution
    method          : str
    notes           : str = ""

    @property
    def delta_t_days(self) -> float:
        return self.delta_t / 86400.0

    @property
    def delta_t_years(self) -> float:
        return self.delta_t / (86400.0 * 365.25)

    @property
    def delta_t_microseconds(self) -> float:
        return self.delta_t * 1e6

    def summary(self) -> str:
        lines = [
            "=" * 58,
            "  NGO — Null Geodesic Observer   |   Chirag Rathi",
            "=" * 58,
            f"  Method         : {self.method}",
            f"  Travel time 1  : {self.t1:.6e} s",
            f"  Travel time 2  : {self.t2:.6e} s",
            f"  Δt             : {self.delta_t:.6e} s",
            f"  Δt             : {self.delta_t_days:.4f} days",
            f"  Δt             : {self.delta_t_microseconds:.4f} μs",
            f"  Coord dist 1   : {self.d1:.4e} m",
            f"  Coord dist 2   : {self.d2:.4e} m",
            f"  Proper dist 1  : {self.proper_d1:.4e} m",
            f"  Proper dist 2  : {self.proper_d2:.4e} m",
            f"  Symmetric?     : {self.is_symmetric}",
            f"  Asymmetry param: {self.asymmetry_param:.4e}",
            "-" * 58,
            f"  Notes: {self.notes}",
            "=" * 58,
        ]
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"DeltaTResult(Δt={self.delta_t:.4e} s, "
            f"symmetric={self.is_symmetric})"
        )


class DeltaTComputer:
    """
    Computes Δt = t₁ − t₂ for two null geodesics.

    Parameters
    ----------
    metric  : Metric — spacetime metric
    solver  : GeodesicSolver (optional, created from metric if not given)

    Example
    -------
    >>> from ngo.core.metric import WeakFieldMetric
    >>> from ngo.core.delta_t import DeltaTComputer
    >>> import numpy as np
    >>>
    >>> # Single mass (Sun) at origin
    >>> metric = WeakFieldMetric(masses=[(2e30, np.array([0., 0., 0.]))])
    >>> computer = DeltaTComputer(metric)
    >>>
    >>> # Two sources at equal coordinate distance, different directions
    >>> result = computer.compute(
    ...     x0_1 = np.array([0., -1e11,  1e10, 0.]),  # source 1
    ...     k0_1 = np.array([1.,  1.0,   0.0,  0.]),  # direction 1
    ...     x0_2 = np.array([0., -1e11, -1e10, 0.]),  # source 2
    ...     k0_2 = np.array([1.,  1.0,   0.0,  0.]),  # direction 2
    ...     lam_end = 1e3
    ... )
    >>> print(result.summary())
    """

    def __init__(self, metric: Metric, solver: Optional[GeodesicSolver] = None):
        self.metric = metric
        self.solver = solver or GeodesicSolver(metric)

    def compute(
        self,
        x0_1    : np.ndarray,
        k0_1    : np.ndarray,
        x0_2    : np.ndarray,
        k0_2    : np.ndarray,
        lam_end : float,
        n_eval  : int = 2000,
        method  : str = 'auto'
    ) -> DeltaTResult:
        """
        Compute Δt between two null geodesics.

        Parameters
        ----------
        x0_1, x0_2 : np.ndarray (4,) — initial positions of source 1, 2
        k0_1, k0_2 : np.ndarray (4,) — initial 4-momenta
        lam_end     : float — affine parameter end value
        n_eval      : int   — integration resolution
        method      : str   — 'general' | 'weak_field' | 'auto'

        Returns
        -------
        DeltaTResult
        """
        # auto-select method
        if method == 'auto':
            method = 'weak_field' if isinstance(self.metric, WeakFieldMetric) \
                     else 'general'

        # solve geodesics
        sol1, sol2 = self.solver.solve_pair(
            x0_1, k0_1, x0_2, k0_2, lam_end, n_eval=n_eval
        )

        # compute path integrals
        if method == 'weak_field' and isinstance(self.metric, WeakFieldMetric):
            pi1 = compute_weak_field_integral(sol1, self.metric)
            pi2 = compute_weak_field_integral(sol2, self.metric)
        else:
            pi1 = compute_path_integral(sol1)
            pi2 = compute_path_integral(sol2)

        t1 = pi1['t_travel']
        t2 = pi2['t_travel']
        L1 = pi1['L'] if 'L' in pi1 else pi1['L_corrected']
        L2 = pi2['L'] if 'L' in pi2 else pi2['L_corrected']

        # coordinate distances (at observer, t=const slice)
        obs_pos = sol1.position_end
        d1 = float(np.linalg.norm(x0_1[1:] - obs_pos[1:]))
        d2 = float(np.linalg.norm(x0_2[1:] - obs_pos[1:]))

        # proper distances
        pd1 = compute_proper_distance(self.metric, x0_1, obs_pos)
        pd2 = compute_proper_distance(self.metric, x0_2, obs_pos)

        # asymmetry check
        asym = check_asymmetry(pi1, pi2)

        # notes
        notes = self._interpret(t1, t2, asym['is_symmetric'])

        return DeltaTResult(
            delta_t         = t1 - t2,
            t1              = t1,
            t2              = t2,
            L1              = L1,
            L2              = L2,
            d1              = d1,
            d2              = d2,
            proper_d1       = pd1,
            proper_d2       = pd2,
            is_symmetric    = asym['is_symmetric'],
            asymmetry_param = asym['asymmetry_parameter'],
            geodesic_1      = sol1,
            geodesic_2      = sol2,
            method          = method,
            notes           = notes
        )

    def _interpret(self, t1: float, t2: float, symmetric: bool) -> str:
        dt = abs(t1 - t2)
        if symmetric:
            return (
                "Symmetric configuration detected: Δt ≈ 0. "
                "Geodesics traverse equal integrated curvature. "
                "This is the counterexample case noted by the reviewer."
            )
        if dt < 1e-6:
            return f"Weak asymmetry: Δt = {dt*1e6:.4f} μs (sub-microsecond regime)."
        if dt < 86400:
            return f"Measurable asymmetry: Δt = {dt:.2f} s = {dt/3600:.4f} hours."
        return f"Large asymmetry: Δt = {dt/86400:.2f} days."


# ── analytic formulae for validation ──────────────────────
def shapiro_delay_analytic(
    M      : float,
    r_emit : float,
    r_obs  : float,
    b      : float
) -> float:
    """
    Analytic Shapiro delay formula (Shapiro 1964).

    Δt_Shapiro = (2GM/c³) × ln(4 r_emit r_obs / b²)

    Parameters
    ----------
    M      : float — mass [kg]
    r_emit : float — distance from mass to emitter [m]
    r_obs  : float — distance from mass to observer [m]
    b      : float — closest approach (impact parameter) [m]

    Returns
    -------
    float — time delay in seconds
    """
    return (2 * G * M / C**3) * np.log(4 * r_emit * r_obs / b**2)


def lensing_delay_analytic(
    M  : float,
    b1 : float,
    b2 : float
) -> float:
    """
    Gravitational lensing time delay between two images (Refsdal 1964).

    Δt_lens = (4GM/c³) × ln(b₂/b₁)

    Parameters
    ----------
    M  : float — lens mass [kg]
    b1 : float — impact parameter of image 1 [m]
    b2 : float — impact parameter of image 2 [m]

    Returns
    -------
    float — time delay in seconds
    """
    return (4 * G * M / C**3) * np.log(b2 / b1)
