# Changelog — NGO (Null Geodesic Observer)
**Author:** Chirag Rathi

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com).

---

## [0.1.0] — April 2026 — Initial Release

### Added
**Core physics engine (`ngo/core/`)**
- `metric.py` — Minkowski, Schwarzschild, WeakField, Custom metrics with Christoffel symbol computation
- `geodesic.py` — Null geodesic solver via `scipy.integrate.solve_ivp` (RK45); null condition enforcement
- `path_integral.py` — Optical path length L = ∫n(x)dσ; effective refractive index n(x) = √(−g₀₀)
- `delta_t.py` — Δt = t₁ − t₂ computation; analytic Shapiro and Refsdal formulae
- `proper_distance.py` — Synge world function approximation for invariant source distance
- `asymmetry_check.py` — Asymmetry measure κ ∈ [0,1]; Killing-vector symmetric case detection

**Observational database (`ngo/database/`)**
- `known_systems.py` — 5 pre-loaded confirmed systems: Q0957+561, SN Refsdal, HE0435-1223, B0218+357, Cassini 2003
- `collector.py` — SQLite database for user-submitted observations; export to JSON

**Analysis tools (`ngo/analysis/`)**
- `strong_field.py` — Exact Schwarzschild Δt; regime scan weak→strong field; photon sphere detection
- `compare.py` — Predicted vs observed Δt comparison table

**Output and visualisation (`ngo/output/`)**
- `plots.py` — Matplotlib: geodesic pair, Δt vs b, known systems, refractive index, null violation
- `export.py` — CSV, JSON, LaTeX table export

**Tests**
- 15 unit tests covering all core physics modules; 15/15 passing

**Paper companion**
- Draft 3 of "Null Geodesic Path Integrals and Observational Signatures of Spacetime Curvature"
- Submitted to Open Journal of Astrophysics, April 2026

### Physics validated against
- Cassini Shapiro delay: predicted 200–300 μs, observed 246.0 μs ✓
- Q0957+561 lensing: observed 417 days confirmed in database ✓
- Symmetric b₁=b₂ configuration: Δt = 0 exactly ✓
- Minkowski Christoffel symbols: all zero ✓
- Schwarzschild far-field limit: recovers Minkowski ✓

---

## [Planned] — v0.2.0

- Full geodesic deviation implementation for Synge world function in strong fields
- Kerr metric (rotating black hole)
- FLRW metric for cosmological light travel time
- Interactive Plotly dashboard for observation submission
- arXiv preprint submission support
