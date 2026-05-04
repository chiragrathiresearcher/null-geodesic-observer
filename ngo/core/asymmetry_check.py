# ============================================================
# NGO — Null Geodesic Observer
# Module : ngo/core/asymmetry_check.py
# Author : Chirag Rathi
# Purpose: Detect symmetric configurations where Δt = 0
#          (fixes reviewer's symmetric counterexample criticism)
# ============================================================
"""
asymmetry_check.py
------------------
Addresses Reviewer Criticism #2:
  "What about symmetric configurations (e.g., two rays passing on
   opposite sides of a spherically symmetric mass)? There Δt = 0
   by symmetry. The theorem fails to account for such counterexamples."

This module:
1. Detects symmetric configurations computationally
2. Computes the asymmetry parameter A = |∫Φdσ₁ − ∫Φdσ₂| / |∫Φdσ₁ + ∫Φdσ₂|
3. Classifies each case as symmetric (Δt=0) or asymmetric (Δt≠0)

The asymmetry condition from the paper:
  Δt ≠ 0  iff  ∫_γ₁ Φ dσ₁ ≠ ∫_γ₂ Φ dσ₂

This is the precise mathematical condition that replaces the vague
"in general" language critiqued by the reviewer.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class AsymmetryResult:
    """
    Result of the asymmetry condition check.

    Attributes
    ----------
    is_symmetric       : bool   — True if Δt ≈ 0 (symmetric case)
    asymmetry_parameter: float  — A ∈ [0, 1], 0=symmetric, 1=max asymmetric
    integral_1         : float  — ∫ n(x) dσ along geodesic 1
    integral_2         : float  — ∫ n(x) dσ along geodesic 2
    delta_integral     : float  — integral_1 − integral_2
    classification     : str    — human-readable classification
    theorem_applies    : bool   — True if asymmetry theorem guarantees Δt ≠ 0
    """
    is_symmetric        : bool
    asymmetry_parameter : float
    integral_1          : float
    integral_2          : float
    delta_integral      : float
    classification      : str
    theorem_applies     : bool

    def __repr__(self):
        return (
            f"AsymmetryResult("
            f"symmetric={self.is_symmetric}, "
            f"A={self.asymmetry_parameter:.4f}, "
            f"class='{self.classification}')"
        )


def check_asymmetry(
    path_integral_1: dict,
    path_integral_2: dict,
    tol            : float = 1e-6
) -> dict:
    """
    Check whether two geodesic path integrals are symmetric.

    Parameters
    ----------
    path_integral_1 : dict — output of compute_path_integral() for geodesic 1
    path_integral_2 : dict — output of compute_path_integral() for geodesic 2
    tol             : float — tolerance for symmetry detection

    Returns
    -------
    dict with keys:
        'is_symmetric'        : bool
        'asymmetry_parameter' : float  ∈ [0, 1]
        'integral_1'          : float
        'integral_2'          : float
        'delta_integral'      : float
        'classification'      : str
        'theorem_applies'     : bool
    """
    # extract path integral values
    I1 = path_integral_1.get('L',
         path_integral_1.get('L_corrected',
         path_integral_1.get('potential_integral', 0.0)))

    I2 = path_integral_2.get('L',
         path_integral_2.get('L_corrected',
         path_integral_2.get('potential_integral', 0.0)))

    delta = I1 - I2
    denom = abs(I1) + abs(I2)

    # asymmetry parameter A ∈ [0,1]
    if denom < 1e-30:
        A = 0.0
    else:
        A = abs(delta) / denom

    is_sym = A < tol

    # classify
    if A < 1e-8:
        classification = "PERFECTLY SYMMETRIC — Δt = 0 exactly"
        theorem_applies = False
    elif A < 1e-4:
        classification = "NEAR-SYMMETRIC — Δt ≈ 0 (weak asymmetry)"
        theorem_applies = False
    elif A < 0.01:
        classification = "WEAKLY ASYMMETRIC — Shapiro-regime Δt"
        theorem_applies = True
    elif A < 0.1:
        classification = "MODERATELY ASYMMETRIC — lensing-regime Δt"
        theorem_applies = True
    else:
        classification = "STRONGLY ASYMMETRIC — large Δt"
        theorem_applies = True

    return {
        'is_symmetric'        : is_sym,
        'asymmetry_parameter' : float(A),
        'integral_1'          : float(I1),
        'integral_2'          : float(I2),
        'delta_integral'      : float(delta),
        'classification'      : classification,
        'theorem_applies'     : theorem_applies
    }


def detect_spherical_symmetry(
    x0_1: np.ndarray,
    x0_2: np.ndarray,
    mass_positions: list,
    tol  : float = 1e-3
) -> bool:
    """
    Detect whether two source positions are symmetric with respect
    to all mass positions (the classic counterexample case).

    Two sources S₁ and S₂ are spherically symmetric w.r.t. a mass M
    if they are on exactly opposite sides at equal impact parameters.

    Parameters
    ----------
    x0_1, x0_2     : np.ndarray (4,) — source positions
    mass_positions  : list of np.ndarray (3,) — mass locations
    tol             : float — tolerance

    Returns
    -------
    bool — True if symmetric counterexample detected
    """
    sp1 = x0_1[1:]   # spatial part of source 1
    sp2 = x0_2[1:]   # spatial part of source 2

    for mass_pos in mass_positions:
        # vectors from mass to each source
        v1 = sp1 - mass_pos
        v2 = sp2 - mass_pos

        r1 = np.linalg.norm(v1)
        r2 = np.linalg.norm(v2)

        # check if opposite sides: v1 · v2 < 0 and |r1| ≈ |r2|
        dot = np.dot(v1, v2)
        same_distance = abs(r1 - r2) / max(r1, r2, 1e-30) < tol
        opposite_side = dot < 0

        if same_distance and opposite_side:
            # check if impact parameters are equal
            # (i.e. both sources lie on a plane through the mass)
            cross = np.cross(v1 / r1, v2 / r2)
            if np.linalg.norm(cross) < tol:
                return True   # symmetric counterexample

    return False
