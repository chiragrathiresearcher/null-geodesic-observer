# ============================================================
# NGO — Null Geodesic Observer
# Module : ngo/core/metric.py
# Author : Chirag Rathi
# Purpose: Define spacetime metrics g_μν as callable objects
# ============================================================
"""
metric.py
---------
Provides metric tensor classes for different spacetimes.

Each metric exposes:
    g(x)        → 4x4 numpy array g_μν at position x
    g_inv(x)    → 4x4 numpy array g^μν (inverse metric)
    christoffel(x) → (4,4,4) array Γ^μ_αβ at position x

Coordinates: x = [t, r, θ, φ] or [t, x, y, z] depending on metric.
"""

import numpy as np
from abc import ABC, abstractmethod


# ── constants ──────────────────────────────────────────────
C  = 2.998e8          # speed of light   [m/s]
G  = 6.674e-11        # gravitational constant [m³/kg/s²]
M_SUN = 1.989e30      # solar mass       [kg]


# ── base class ─────────────────────────────────────────────
class Metric(ABC):
    """Abstract base class for all spacetime metrics."""

    @abstractmethod
    def g(self, x: np.ndarray) -> np.ndarray:
        """
        Metric tensor g_μν at position x.

        Parameters
        ----------
        x : np.ndarray, shape (4,)
            Coordinates [x0, x1, x2, x3]

        Returns
        -------
        np.ndarray, shape (4, 4)
            Covariant metric tensor g_μν
        """

    def g_inv(self, x: np.ndarray) -> np.ndarray:
        """Inverse metric g^μν at position x."""
        return np.linalg.inv(self.g(x))

    def christoffel(self, x: np.ndarray, dx: float = 1e-6) -> np.ndarray:
        """
        Christoffel symbols Γ^μ_αβ computed by numerical differentiation.

        Γ^μ_αβ = ½ g^μσ (∂_α g_βσ + ∂_β g_ασ − ∂_σ g_αβ)

        Parameters
        ----------
        x  : np.ndarray, shape (4,)  — coordinates
        dx : float — finite difference step

        Returns
        -------
        np.ndarray, shape (4, 4, 4)
            Γ[mu, alpha, beta]
        """
        n = 4
        g0    = self.g(x)
        g_inv = np.linalg.inv(g0)

        # partial derivatives ∂_σ g_μν  →  dg[sigma, mu, nu]
        dg = np.zeros((n, n, n))
        for sigma in range(n):
            xp = x.copy(); xp[sigma] += dx
            xm = x.copy(); xm[sigma] -= dx
            dg[sigma] = (self.g(xp) - self.g(xm)) / (2 * dx)

        # Γ^μ_αβ = ½ g^μσ (∂_α g_βσ + ∂_β g_ασ − ∂_σ g_αβ)
        Gamma = np.zeros((n, n, n))
        for mu in range(n):
            for alpha in range(n):
                for beta in range(n):
                    s = 0.0
                    for sigma in range(n):
                        s += g_inv[mu, sigma] * (
                            dg[alpha, beta, sigma]
                            + dg[beta, alpha, sigma]
                            - dg[sigma, alpha, beta]
                        )
                    Gamma[mu, alpha, beta] = 0.5 * s

        return Gamma

    def null_check(self, x: np.ndarray, k: np.ndarray) -> float:
        """
        Check the null condition g_μν k^μ k^ν = 0.

        Returns the value (should be ~0 for a photon).
        """
        g = self.g(x)
        return float(np.einsum('ij,i,j', g, k, k))

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# ══════════════════════════════════════════════════════════════
# METRIC 1 — Minkowski (flat spacetime, Case I from notes)
# ══════════════════════════════════════════════════════════════
class MinkowskiMetric(Metric):
    """
    Flat Minkowski spacetime in Cartesian coordinates.
    Coordinates: x = [ct, x, y, z]

    ds² = −c²dt² + dx² + dy² + dz²
    g_μν = diag(−1, +1, +1, +1)

    This is Case I from Chirag Rathi's notes:
    straight parallel geodesics, d₁=d₂ → t₁=t₂ exactly.
    """

    def g(self, x: np.ndarray) -> np.ndarray:
        return np.diag([-1.0, 1.0, 1.0, 1.0])

    def christoffel(self, x: np.ndarray, dx: float = 1e-6) -> np.ndarray:
        # Flat spacetime: all Christoffel symbols are exactly zero
        return np.zeros((4, 4, 4))

    def __repr__(self):
        return "MinkowskiMetric(signature=(-,+,+,+))"


# ══════════════════════════════════════════════════════════════
# METRIC 2 — Schwarzschild (single spherical mass)
# ══════════════════════════════════════════════════════════════
class SchwarzschildMetric(Metric):
    """
    Schwarzschild metric for a spherical mass M.
    Coordinates: x = [t, r, θ, φ]   (geometric units: c=G=1 internally)

    ds² = −(1 − r_s/r)c²dt² + (1 − r_s/r)⁻¹ dr² + r²dθ² + r²sin²θ dφ²

    where r_s = 2GM/c² is the Schwarzschild radius.

    Parameters
    ----------
    M : float — mass in kg (default: solar mass)
    """

    def __init__(self, M: float = M_SUN):
        self.M  = M
        self.rs = 2 * G * M / C**2   # Schwarzschild radius [m]

    def g(self, x: np.ndarray) -> np.ndarray:
        r   = x[1]
        th  = x[2]

        if r <= self.rs:
            raise ValueError(
                f"r={r:.3e} m is inside or at the Schwarzschild radius "
                f"r_s={self.rs:.3e} m. Geodesic integration invalid here."
            )

        f   = 1.0 - self.rs / r
        sin2 = np.sin(th)**2

        return np.diag([
            -f * C**2,          # g_tt
             1.0 / f,           # g_rr
             r**2,              # g_θθ
             r**2 * sin2        # g_φφ
        ])

    def __repr__(self):
        return f"SchwarzschildMetric(M={self.M:.3e} kg, r_s={self.rs:.3e} m)"


# ══════════════════════════════════════════════════════════════
# METRIC 3 — Weak Field / Post-Newtonian (for Shapiro delay)
# ══════════════════════════════════════════════════════════════
class WeakFieldMetric(Metric):
    """
    Weak-field (post-Newtonian) metric for a collection of point masses.
    Coordinates: x = [ct, x, y, z]  (Cartesian)

    ds² = −(1 + 2Φ/c²)c²dt² + (1 − 2Φ/c²)(dx² + dy² + dz²)

    where Φ(x) = −Σ GM_i / |x − x_i|  is the Newtonian potential.

    This is the metric underlying equation (3.6) in the paper.

    Parameters
    ----------
    masses : list of (M, position) tuples
        M        — mass in kg
        position — np.ndarray shape (3,) in metres [x, y, z]
    """

    def __init__(self, masses: list):
        self.masses = masses   # [(M1, pos1), (M2, pos2), ...]

    def potential(self, x: np.ndarray) -> float:
        """Newtonian gravitational potential Φ(x) at spatial position x[1:4]."""
        pos = x[1:4]
        phi = 0.0
        for M, source_pos in self.masses:
            r = np.linalg.norm(pos - source_pos)
            if r < 1e3:   # avoid singularity; 1 km minimum
                r = 1e3
            phi -= G * M / r
        return phi

    def g(self, x: np.ndarray) -> np.ndarray:
        phi  = self.potential(x)
        f_tt = -(1.0 + 2.0 * phi / C**2)
        f_sp =  (1.0 - 2.0 * phi / C**2)

        return np.diag([f_tt, f_sp, f_sp, f_sp])

    def __repr__(self):
        return f"WeakFieldMetric(n_masses={len(self.masses)})"


# ══════════════════════════════════════════════════════════════
# METRIC 4 — Custom user-defined metric
# ══════════════════════════════════════════════════════════════
class CustomMetric(Metric):
    """
    User-defined metric tensor.

    Parameters
    ----------
    g_func : callable
        Function g_func(x) → np.ndarray shape (4,4)
        x is np.ndarray shape (4,) of coordinates.

    Example
    -------
    >>> def my_metric(x):
    ...     return np.diag([-1.0, 1.0, 1.0, 1.0])   # flat
    >>> m = CustomMetric(my_metric)
    """

    def __init__(self, g_func):
        if not callable(g_func):
            raise TypeError("g_func must be callable: g_func(x) → np.ndarray (4,4)")
        self._g_func = g_func

    def g(self, x: np.ndarray) -> np.ndarray:
        result = self._g_func(x)
        result = np.asarray(result, dtype=float)
        if result.shape != (4, 4):
            raise ValueError(f"g_func must return shape (4,4), got {result.shape}")
        return result

    def __repr__(self):
        return f"CustomMetric(g_func={self._g_func.__name__})"


# ── convenience factory ────────────────────────────────────
def get_metric(name: str, **kwargs) -> Metric:
    """
    Convenience factory to get a metric by name.

    Parameters
    ----------
    name : str
        One of: 'minkowski', 'schwarzschild', 'weak_field', 'custom'
    **kwargs
        Passed to the metric constructor.

    Returns
    -------
    Metric instance

    Examples
    --------
    >>> m = get_metric('schwarzschild', M=2e30)
    >>> m = get_metric('minkowski')
    """
    registry = {
        'minkowski'    : MinkowskiMetric,
        'schwarzschild': SchwarzschildMetric,
        'weak_field'   : WeakFieldMetric,
        'custom'       : CustomMetric,
    }
    name = name.lower().strip()
    if name not in registry:
        raise ValueError(
            f"Unknown metric '{name}'. "
            f"Available: {list(registry.keys())}"
        )
    return registry[name](**kwargs)
