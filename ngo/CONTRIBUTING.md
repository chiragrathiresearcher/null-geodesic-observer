# Contributing to NGO — Null Geodesic Observer

**Author:** Chirag Rathi  
**Repo:** github.com/ChiragRathi/null-geodesic-observer

Thank you for your interest in contributing. NGO is an open science project — contributions from physicists, astronomers, and developers are welcome.

---

## Ways to Contribute

### 1. Submit an Observation
The most valuable contribution is adding a real measurement to the database.

```python
from ngo.database.collector import ObservationDB, Observation

db = ObservationDB()
db.submit(Observation(
    system_name    = 'YOUR_SYSTEM',
    obs_type       = 'lensing',        # or 'shapiro', 'computed', 'custom'
    delta_t_s      = 100 * 86400,      # Δt in seconds
    delta_t_err_s  = 5   * 86400,      # uncertainty in seconds
    reference      = 'Author+ Year, Journal, Vol, Page',
    notes          = 'Brief description of the system',
    submitted_by   = 'Your Name'
))
```

Then open a Pull Request with the updated `data/observations.db`.

---

### 2. Add a New Metric

Add a new spacetime metric by subclassing `Metric` in `ngo/core/metric.py`:

```python
class KerrMetric(Metric):
    """
    Kerr metric for a rotating black hole.
    Coordinates: x = [t, r, θ, φ]
    """
    def __init__(self, M: float, a: float):
        self.M = M
        self.a = a  # spin parameter  0 ≤ a ≤ GM/c²

    def g(self, x: np.ndarray) -> np.ndarray:
        # implement g_μν for Kerr
        ...
```

---

### 3. Improve the Geodesic Solver

Current limitations:
- Breaks down at `b/r_s < 3` (near photon sphere)
- World function uses straight-line approximation for weak fields
- No symplectic integration option

PRs improving these are especially welcome.

---

### 4. Report a Bug or Physics Error

Open a GitHub Issue with:
- The metric you were using
- Initial conditions `x0`, `k0`
- Expected vs actual output
- NGO version (`ngo.__version__`)

---

## Code Standards

- **Python 3.10+**
- **Type hints** on all function signatures
- **Docstrings** in NumPy style
- **Author line** in every new file: `# Author: Your Name`
- **Units** always stated in docstrings (SI: metres, kilograms, seconds)
- Run existing tests before submitting: `python -m pytest tests/`

---

## Pull Request Checklist

- [ ] New physics: cite the relevant equation/paper in the docstring
- [ ] New module: add to the appropriate `__init__.py`
- [ ] New system: add to `ngo/database/known_systems.py` with full reference
- [ ] All existing tests pass
- [ ] New tests added for new functionality

---

## Academic Credit

If NGO contributes to your published research, please cite:

```
Rathi, C. (2026). Null Geodesic Path Integrals and Observational
Signatures of Spacetime Curvature: From Shapiro Delay to
Gravitational Lensing. Open Journal of Astrophysics [submitted].
NGO software: github.com/ChiragRathi/null-geodesic-observer
```

Contributors who add confirmed observational data or significant new physics
will be acknowledged in the paper's Acknowledgements section.

---

## Contact

Open a GitHub Issue for questions.  
All contributions are reviewed by Chirag Rathi.
