# ============================================================
# NGO — Null Geodesic Observer
# Module : ngo/core/geodesic.py
# Author : Chirag Rathi
# Purpose: Numerically solve null geodesic equations (2.3)(2.4)
# ============================================================
"""
geodesic.py
-----------
Solves the null geodesic equation:

    dk^μ/dλ + Γ^μ_αβ k^α k^β = 0        (equation 2.4)
    g_μν k^μ k^ν = 0                      (equation 2.3, null condition)

using scipy.integrate.solve_ivp with RK45 adaptive stepping.

The state vector is:
    y = [x^0, x^1, x^2, x^3, k^0, k^1, k^2, k^3]
       = [position (4), tangent vector (4)]

Output is a GeodesicSolution object containing the full path γ(λ).
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional

from .metric import Metric


# ── result container ───────────────────────────────────────
@dataclass
class GeodesicSolution:
    """
    Contains the result of a null geodesic integration.

    Attributes
    ----------
    x        : np.ndarray (4, N) — spacetime coordinates along path
    k        : np.ndarray (4, N) — tangent vector (photon 4-momentum)
    lam      : np.ndarray (N,)   — affine parameter values
    success  : bool              — did the integration converge?
    message  : str               — solver message
    metric   : Metric            — the metric used
    null_violation : np.ndarray  — g_μν k^μ k^ν along path (should be ~0)
    """
    x             : np.ndarray
    k             : np.ndarray
    lam           : np.ndarray
    success       : bool
    message       : str
    metric        : Metric
    null_violation: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def position_start(self) -> np.ndarray:
        return self.x[:, 0]

    @property
    def position_end(self) -> np.ndarray:
        return self.x[:, -1]

    @property
    def n_steps(self) -> int:
        return self.lam.shape[0]

    def spatial_path(self) -> np.ndarray:
        """Return spatial coordinates only: shape (3, N) = [x1, x2, x3]."""
        return self.x[1:, :]

    def max_null_violation(self) -> float:
        """Maximum |g_μν k^μ k^ν| along the path — quality indicator."""
        if len(self.null_violation) == 0:
            return float('nan')
        return float(np.max(np.abs(self.null_violation)))

    def __repr__(self):
        return (
            f"GeodesicSolution("
            f"steps={self.n_steps}, "
            f"success={self.success}, "
            f"max_null_violation={self.max_null_violation():.2e})"
        )


# ── geodesic solver ────────────────────────────────────────
class GeodesicSolver:
    """
    Solves the null geodesic equation for a given metric.

    Parameters
    ----------
    metric   : Metric  — spacetime metric object from ngo.core.metric
    rtol     : float   — relative tolerance for ODE solver (default 1e-9)
    atol     : float   — absolute tolerance for ODE solver (default 1e-12)
    max_step : float   — maximum affine parameter step size (None = no limit)

    Example
    -------
    >>> from ngo.core.metric import SchwarzschildMetric
    >>> from ngo.core.geodesic import GeodesicSolver
    >>>
    >>> metric = SchwarzschildMetric(M=2e30)
    >>> solver = GeodesicSolver(metric)
    >>>
    >>> x0 = np.array([0.0, 1e11, np.pi/2, 0.0])   # start position
    >>> k0 = np.array([1.0, 0.0, 0.0, 1e-11])       # initial 4-momentum
    >>> sol = solver.solve(x0, k0, lam_end=1e3)
    """

    def __init__(
        self,
        metric   : Metric,
        rtol     : float = 1e-9,
        atol     : float = 1e-12,
        max_step : Optional[float] = None
    ):
        self.metric   = metric
        self.rtol     = rtol
        self.atol     = atol
        self.max_step = max_step

    def _geodesic_rhs(self, lam: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the geodesic ODE system.

        State: y = [x^0, x^1, x^2, x^3, k^0, k^1, k^2, k^3]

        dx^μ/dλ = k^μ
        dk^μ/dλ = −Γ^μ_αβ k^α k^β
        """
        x = y[:4]
        k = y[4:]

        try:
            Gamma = self.metric.christoffel(x)
        except Exception as e:
            # If metric fails (e.g. metric undefined at this x, e.g. inside horizon),
            # propagate the error so the integrator halts and caller can handle it.
            raise RuntimeError(
                "Metric.christoffel evaluation failed during geodesic RHS "
                f"evaluation at x={x}. Aborting integration. Original error: {e}"
            )

        # dk^μ/dλ = −Γ^μ_αβ k^α k^β
        dk = -np.einsum('mab,a,b->m', Gamma, k, k)

        return np.concatenate([k, dk])

    def _normalize_null(self, x0: np.ndarray, k0: np.ndarray) -> np.ndarray:
        """
        Adjust k^0 so that the null condition g_μν k^μ k^ν = 0 is satisfied.
        Solves: g_00 (k^0)² + 2 g_0i k^0 k^i + g_ij k^i k^j = 0

        This function now handles the degenerate quadratic case when g_00 ≈ 0
        (linear equation in k^0) robustly.
        """
        g   = self.metric.g(x0)
        k_s = k0[1:]   # spatial components

        # coefficients of quadratic in k^0
        A = float(g[0, 0])
        B = float(2.0 * np.dot(g[0, 1:], k_s))
        C_coeff = float(np.einsum('ij,i,j', g[1:, 1:], k_s, k_s))

        eps = 1e-30
        # Handle near-degenerate quadratic (A ≈ 0): treat as linear B*k0 + C = 0
        if abs(A) < eps:
            if abs(B) < eps:
                raise ValueError(
                    "Degenerate null-condition: both A and B coefficients are ~0;"
                    " cannot determine k^0 from given spatial momentum."
                )
            k0_new = -C_coeff / B
            k_normalized = k0.copy()
            k_normalized[0] = k0_new
            return k_normalized

        discriminant = B**2 - 4 * A * C_coeff
        if discriminant < 0:
            raise ValueError(
                "Cannot satisfy null condition with given spatial momentum: "
                "quadratic has no real roots. Check initial conditions."
            )

        sqrt_disc = np.sqrt(discriminant)
        # two candidate roots
        k0_opt1 = (-B + sqrt_disc) / (2 * A)
        k0_opt2 = (-B - sqrt_disc) / (2 * A)

        # pick the root consistent with forward time propagation (k^0 > 0)
        candidates = [k0_opt1, k0_opt2]
        k0_new = None
        for cand in candidates:
            if np.isfinite(cand) and cand > 0:
                k0_new = cand
                break
        if k0_new is None:
            # fall back to the larger (in magnitude) root if neither is >0
            k0_new = max(candidates, key=lambda v: abs(v))

        k_normalized = k0.copy()
        k_normalized[0] = k0_new
        return k_normalized

    def solve(
        self,
        x0          : np.ndarray,
        k0          : np.ndarray,
        lam_end     : float,
        lam_start   : float = 0.0,
        n_eval      : int   = 1000,
        enforce_null: bool  = True,
        events      : Optional[list] = None
    ) -> GeodesicSolution:
        """
        Integrate the null geodesic from λ=lam_start to λ=lam_end.

        Parameters
        ----------
        x0           : np.ndarray (4,) — initial spacetime position
        k0           : np.ndarray (4,) — initial photon 4-momentum
        lam_end      : float — final affine parameter value
        lam_start    : float — initial affine parameter (default 0)
        n_eval       : int   — number of output points
        enforce_null : bool  — auto-correct k^0 to satisfy null condition
        events       : list  — scipy event functions for early termination

        Returns
        -------
        GeodesicSolution
        """
        x0 = np.asarray(x0, dtype=float)
        k0 = np.asarray(k0, dtype=float)

        if x0.shape != (4,) or k0.shape != (4,):
            raise ValueError("x0 and k0 must each have shape (4,)")

        # enforce null condition
        if enforce_null:
            k0 = self._normalize_null(x0, k0)

        y0   = np.concatenate([x0, k0])
        lam_eval = np.linspace(lam_start, lam_end, n_eval)

        ivp_kwargs = dict(
            fun     = self._geodesic_rhs,
            t_span  = (lam_start, lam_end),
            y0      = y0,
            method  = 'RK45',
            t_eval  = lam_eval,
            rtol    = self.rtol,
            atol    = self.atol,
            events  = events,
        )
        if self.max_step is not None:
            ivp_kwargs['max_step'] = self.max_step

        result = solve_ivp(**ivp_kwargs)

        x_path = result.y[:4, :]
        k_path = result.y[4:, :]

        # compute null violation along path
        null_viol = np.array([
            self.metric.null_check(x_path[:, i], k_path[:, i])
            for i in range(x_path.shape[1])
        ])

        return GeodesicSolution(
            x             = x_path,
            k             = k_path,
            lam           = result.t,
            success       = result.success,
            message       = result.message,
            metric        = self.metric,
            null_violation= null_viol
        )

    def solve_pair(
        self,
        x0_1    : np.ndarray,
        k0_1    : np.ndarray,
        x0_2    : np.ndarray,
        k0_2    : np.ndarray,
        lam_end : float,
        **kwargs
    ) -> tuple:
        """
        Solve two null geodesics (for two sources) simultaneously.

        Returns
        -------
        (GeodesicSolution, GeodesicSolution)
        """
        sol1 = self.solve(x0_1, k0_1, lam_end, **kwargs)
        sol2 = self.solve(x0_2, k0_2, lam_end, **kwargs)
        return sol1, sol2
