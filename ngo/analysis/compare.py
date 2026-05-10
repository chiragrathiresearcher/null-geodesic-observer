# ============================================================
# NGO — Null Geodesic Observer
# Module : ngo/analysis/compare.py
# Author : Chirag Rathi
# Purpose: Compare NGO predictions against known observations
# ============================================================
"""
compare.py
----------
Validates NGO computed Δt against observed values from the
known systems database.

For each system:
  - Computes predicted Δt using appropriate metric
  - Compares with observed Δt
  - Reports fractional residual
  - Flags disagreements beyond 3σ
"""

import numpy as np
from ..database.known_systems import KNOWN_SYSTEMS
from ..core.delta_t import shapiro_delay_analytic, lensing_delay_analytic
from ..core.metric import C, G


def compare_shapiro(system: dict) -> dict:
    """Compare predicted vs observed Shapiro delay for a known system."""
    params = system['parameters']
    M      = params.get('M_lens', 1.989e30)
    b      = params.get('b_min_solar_r', 1.6) * 6.957e8   # solar radii to m
    r_emit = 1.5e12    # ~Cassini distance (10 AU)
    r_obs  = 1.5e11    # 1 AU

    predicted = shapiro_delay_analytic(M, r_emit, r_obs, b)
    observed  = system['delta_t_obs']
    error     = system['delta_t_err']

    residual     = predicted - observed
    frac_residual = residual / observed if observed else 0.0
    sigma        = abs(residual) / error if error else 0.0

    return {
        'system'       : system['name'],
        'predicted_s'  : predicted,
        'observed_s'   : observed,
        'error_s'      : error,
        'residual_s'   : residual,
        'frac_residual': frac_residual,
        'sigma'        : sigma,
        'agreement'    : sigma < 3.0,
    }


def compare_all_known() -> list:
    """
    Run comparison for all pre-loaded systems with known Δt.

    Returns
    -------
    list of comparison dicts
    """
    results = []
    for sys in KNOWN_SYSTEMS:
        if sys['delta_t_obs'] is None:
            continue
        if sys['type'] == 'shapiro':
            r = compare_shapiro(sys)
            results.append(r)
        else:
            # For lensing systems, report observed value only
            results.append({
                'system'       : sys['name'],
                'predicted_s'  : None,
                'observed_s'   : sys['delta_t_obs'],
                'error_s'      : sys['delta_t_err'],
                'residual_s'   : None,
                'frac_residual': None,
                'sigma'        : None,
                'agreement'    : None,
                'note'         : 'Lensing: full simulation required for prediction'
            })
    return results


def print_comparison_table(results: list):
    """Pretty-print comparison results."""
    print("=" * 72)
    print(f"  {'System':<20} {'Obs Δt':>14} {'Pred Δt':>14} {'σ':>8}")
    print("=" * 72)
    for r in results:
        obs  = r['observed_s']
        pred = r['predicted_s']
        sig  = r['sigma']

        obs_str  = f"{obs:.4e} s"  if obs  is not None else "—"
        pred_str = f"{pred:.4e} s" if pred is not None else "—"
        sig_str  = f"{sig:.2f}σ"   if sig  is not None else "—"

        flag = " ✓" if r.get('agreement') else (" ✗" if r.get('agreement') is False else "")
        print(f"  {r['system']:<20} {obs_str:>14} {pred_str:>14} {sig_str:>8}{flag}")
    print("=" * 72)
