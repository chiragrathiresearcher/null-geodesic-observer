#!/usr/bin/env python3
# ============================================================
# NGO — Null Geodesic Observer
# File   : demo.py
# Author : Chirag Rathi
# Purpose: Runnable demonstration of the NGO physics engine
#
# Run with: python demo.py
# ============================================================

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ngo.core.metric      import WeakFieldMetric, MinkowskiMetric
from ngo.core.delta_t     import (DeltaTComputer,
                                   shapiro_delay_analytic,
                                   lensing_delay_analytic)
from ngo.core.asymmetry_check import check_asymmetry
from ngo.analysis.strong_field import schwarzschild_shapiro_exact
from ngo.database.known_systems import list_systems, get_system, get_all_delta_t
from ngo.database.collector import ObservationDB, Observation

C   = 2.998e8
G   = 6.674e-11
M_S = 1.989e30

SEP = "=" * 60

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ──────────────────────────────────────────────────────────
# DEMO 1: Shapiro Delay Analytic Formula
# ──────────────────────────────────────────────────────────
section("1. Shapiro Delay — Analytic Formula (Cassini)")

M      = M_S
r_emit = 1.5e12    # ~10 AU (Cassini distance)
r_obs  = 1.5e11    # 1 AU
b      = 1.6 * 6.957e8   # 1.6 solar radii (closest approach)

dt_shapiro = shapiro_delay_analytic(M, r_emit, r_obs, b)
print(f"  Mass          : {M:.3e} kg (1 solar mass)")
print(f"  r_emitter     : {r_emit:.3e} m")
print(f"  r_observer    : {r_obs:.3e} m")
print(f"  Impact param  : {b:.3e} m ({b/6.957e8:.1f} solar radii)")
print(f"\n  Shapiro delay : {dt_shapiro*1e6:.2f} μs")
print(f"  Cassini obs.  : 246.00 μs")
print(f"  Ratio         : {dt_shapiro*1e6/246:.4f}")


# ──────────────────────────────────────────────────────────
# DEMO 2: Lensing Time Delay
# ──────────────────────────────────────────────────────────
section("2. Gravitational Lensing Delay (Q0957+561 analogy)")

M_lens = 1e11 * M_S     # galaxy mass
b1     = 1e22            # impact parameter image 1
b2     = 1.5e22          # impact parameter image 2

dt_lens = lensing_delay_analytic(M_lens, b1, b2)
print(f"  Lens mass     : {M_lens:.2e} kg (~10^11 solar masses)")
print(f"  b₁            : {b1:.2e} m")
print(f"  b₂            : {b2:.2e} m")
print(f"\n  Predicted Δt  : {dt_lens/86400:.1f} days")
print(f"  Q0957 obs.    : 417.0 days")


# ──────────────────────────────────────────────────────────
# DEMO 3: Asymmetry Condition Check
# ──────────────────────────────────────────────────────────
section("3. Asymmetry Condition (Addresses Reviewer Criticism)")

print("\n  Case A: SYMMETRIC — two sources on opposite sides of mass")
pi_sym1 = {'L': 1.0e11}
pi_sym2 = {'L': 1.0e11}
asym_A  = check_asymmetry(pi_sym1, pi_sym2)
print(f"  Asymmetry parameter A = {asym_A['asymmetry_parameter']:.6f}")
print(f"  Classification : {asym_A['classification']}")
print(f"  Theorem applies: {asym_A['theorem_applies']}")

print("\n  Case B: ASYMMETRIC — sources at different curvature paths")
pi_asy1 = {'L': 1.0e11}
pi_asy2 = {'L': 1.05e11}
asym_B  = check_asymmetry(pi_asy1, pi_asy2)
print(f"  Asymmetry parameter A = {asym_B['asymmetry_parameter']:.6f}")
print(f"  Classification : {asym_B['classification']}")
print(f"  Theorem applies: {asym_B['theorem_applies']}")


# ──────────────────────────────────────────────────────────
# DEMO 4: Strong-Field Exact Schwarzschild
# ──────────────────────────────────────────────────────────
section("4. Strong-Field Exact vs Weak-Field Comparison")

r_s = 2 * G * M_S / C**2
print(f"  Schwarzschild radius r_s = {r_s:.2f} m")
print(f"  Photon sphere  r_ph = {1.5*r_s:.2f} m\n")

for b_rs in [1000, 100, 10, 3]:
    b = b_rs * r_s
    try:
        res = schwarzschild_shapiro_exact(M_S, 1e12, 1e11, b)
        print(
            f"  b/r_s={b_rs:5d}  |  "
            f"exact={res['delta_t_exact']*1e6:8.3f} μs  |  "
            f"analytic={res['delta_t_analytic']*1e6:8.3f} μs  |  "
            f"error={res['delta_t_error_frac']*100:.4f}%  |  "
            f"{res['regime']}"
        )
    except Exception as e:
        print(f"  b/r_s={b_rs:5d}  |  Error: {e}")


# ──────────────────────────────────────────────────────────
# DEMO 5: Known Systems Database
# ──────────────────────────────────────────────────────────
section("5. Pre-loaded Observation Database")

all_dt = get_all_delta_t()
print(f"  {'System':<25} {'Δt':>20}")
print(f"  {'-'*25} {'-'*20}")
for name, dt in all_dt.items():
    if dt > 86400:
        print(f"  {name:<25} {dt/86400:>15.1f} days")
    else:
        print(f"  {name:<25} {dt*1e6:>15.1f} μs")


# ──────────────────────────────────────────────────────────
# DEMO 6: Submit Observation to Database
# ──────────────────────────────────────────────────────────
section("6. Submit Observation to SQLite Database")

db = ObservationDB('/tmp/ngo_demo.db')
db.load_known_systems()

obs_id = db.submit(Observation(
    system_name    = 'Demo_System',
    obs_type       = 'computed',
    delta_t_s      = dt_shapiro,
    delta_t_err_s  = 0.0,
    metric_type    = 'weak_field',
    is_symmetric   = False,
    asymmetry_param= 0.0476,
    reference      = 'NGO demo.py',
    notes          = 'Cassini Shapiro delay computed analytically',
    submitted_by   = 'Chirag Rathi'
))
print(f"  Submitted observation ID: {obs_id}")
print(db.summary())


# ──────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  NGO demo complete.")
print("  Author : Chirag Rathi")
print("  Paper  : Null Geodesic Path Integrals and Observational")
print("           Signatures of Spacetime Curvature")
print(f"{SEP}\n")
