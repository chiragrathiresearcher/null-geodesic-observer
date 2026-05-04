# NGO — Null Geodesic Observer

**Author:** Chirag Rathi  
**Version:** 0.1.0  
**License:** MIT

---

## What is NGO?

NGO is a Python package for computing and collecting **light travel time asymmetry** — the phenomenon where two photons from sources at equal distance arrive at an observer at different times due to spacetime curvature.

This is the companion software to the paper:

> *"Null Geodesic Path Integrals and Observational Signatures of Spacetime Curvature: From Shapiro Delay to Cosmological Lensing"*  
> Chirag Rathi (2026)

---

## The Core Physics

In flat spacetime: `d₁ = d₂ → t₁ = t₂` (equal distance = equal travel time)

In curved spacetime: `d₁ = d₂` does **not** imply `t₁ = t₂`

The travel time difference is:

```
Δt = (1/c) × Δ∫ [g_μν k^μ k^ν]^½ dλ
```

NGO computes this integral numerically for any spacetime metric.

---

## Installation

```bash
git clone https://github.com/ChiragRathi/null-geodesic-observer
cd null-geodesic-observer
pip install -e .
```

---

## Quick Start

```python
from ngo.core import WeakFieldMetric, DeltaTComputer
import numpy as np

# Single mass (Sun) at origin
metric = WeakFieldMetric(masses=[(2e30, np.array([0., 0., 0.]))])

# Compute Δt for two sources at equal distance, different paths
computer = DeltaTComputer(metric)
result = computer.compute(
    x0_1    = np.array([0., -1e11,  1e10, 0.]),
    k0_1    = np.array([1.,  1.0,   0.0,  0.]),
    x0_2    = np.array([0., -1e11, -1e10, 0.]),
    k0_2    = np.array([1.,  1.0,   0.0,  0.]),
    lam_end = 1e3
)
print(result.summary())
```

---

## Modules

| Module | Purpose |
|--------|---------|
| `ngo.core.metric` | Define spacetime metrics (Minkowski, Schwarzschild, Weak-field, Custom) |
| `ngo.core.geodesic` | Solve null geodesic equations numerically (RK45) |
| `ngo.core.path_integral` | Compute ∫n(x)dσ along any geodesic |
| `ngo.core.delta_t` | Compute Δt between two null geodesics |
| `ngo.core.proper_distance` | Invariant proper distance calculator |
| `ngo.core.asymmetry_check` | Detect symmetric vs asymmetric configurations |
| `ngo.database.known_systems` | Pre-loaded lensing + Shapiro delay systems |
| `ngo.database.collector` | SQLite database for user observations |

---

## Known Systems Pre-loaded

- Q0957+561 (Twin Quasar) — Δt = 417 days
- B0218+357 — Δt = 10.5 days
- HE0435-1223 — Δt = 14.4 days
- SN Refsdal — Δt = 376 days
- Cassini Shapiro Delay — Δt = 246 μs

---

## Contributing Observations

```python
from ngo.database.collector import ObservationDB, Observation

db = ObservationDB()
db.submit(Observation(
    system_name   = 'Your System',
    obs_type      = 'lensing',
    delta_t_s     = 100 * 86400,
    delta_t_err_s = 5 * 86400,
    reference     = 'Your citation',
    submitted_by  = 'Your name'
))
print(db.summary())
```

---

## Citation

If you use NGO in your research, please cite:

```
Rathi, C. (2026). Null Geodesic Path Integrals and Observational
Signatures of Spacetime Curvature: From Shapiro Delay to
Cosmological Lensing. [Journal TBD]
```

---

## License

MIT License — free to use, modify, and distribute with attribution.
