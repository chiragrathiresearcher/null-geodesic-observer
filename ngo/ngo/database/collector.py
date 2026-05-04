# ============================================================
# NGO — Null Geodesic Observer
# Module : ngo/database/collector.py
# Author : Chirag Rathi
# Purpose: SQLite database for user-submitted observations
# ============================================================
"""
collector.py
------------
Manages the SQLite database for collecting and storing
light travel time asymmetry observations.

Users can submit:
  - Lensing time delay measurements
  - Shapiro delay measurements
  - Custom geodesic path integral results
  - Comparisons with NGO computed predictions

The database file travels with the repository and can be
shared, merged, and versioned on GitHub.
"""

import sqlite3
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List


# default database location
DEFAULT_DB = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'observations.db'
)


@dataclass
class Observation:
    """
    A single light travel time asymmetry observation.

    Fields
    ------
    system_name    : str   — name of the astrophysical system
    obs_type       : str   — 'lensing' | 'shapiro' | 'computed' | 'custom'
    delta_t_s      : float — measured/computed Δt in seconds
    delta_t_err_s  : float — uncertainty in seconds (0 if computed)
    d1_m           : float — distance to source 1 [metres]
    d2_m           : float — distance to source 2 [metres]
    metric_type    : str   — metric used ('schwarzschild'|'weak_field'|etc.)
    is_symmetric   : bool  — was this a symmetric configuration?
    asymmetry_param: float — asymmetry parameter A ∈ [0,1]
    reference      : str   — citation or 'NGO_computed'
    notes          : str   — any additional notes
    submitted_by   : str   — contributor name/handle
    timestamp      : str   — ISO format timestamp
    """
    system_name    : str
    obs_type       : str
    delta_t_s      : float
    delta_t_err_s  : float   = 0.0
    d1_m           : float   = 0.0
    d2_m           : float   = 0.0
    metric_type    : str     = 'unknown'
    is_symmetric   : bool    = False
    asymmetry_param: float   = 0.0
    reference      : str     = ''
    notes          : str     = ''
    submitted_by   : str     = 'anonymous'
    timestamp      : str     = ''

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class ObservationDB:
    """
    SQLite-backed database for NGO observations.

    Parameters
    ----------
    db_path : str — path to SQLite file (created if not exists)

    Example
    -------
    >>> db = ObservationDB()
    >>> db.submit(Observation(
    ...     system_name    = 'Q0957+561',
    ...     obs_type       = 'lensing',
    ...     delta_t_s      = 417 * 86400,
    ...     delta_t_err_s  = 3 * 86400,
    ...     reference      = 'Kundic et al. 1997',
    ...     submitted_by   = 'Chirag Rathi'
    ... ))
    >>> results = db.query(obs_type='lensing')
    """

    CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS observations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        system_name     TEXT    NOT NULL,
        obs_type        TEXT    NOT NULL,
        delta_t_s       REAL    NOT NULL,
        delta_t_err_s   REAL    DEFAULT 0.0,
        d1_m            REAL    DEFAULT 0.0,
        d2_m            REAL    DEFAULT 0.0,
        metric_type     TEXT    DEFAULT 'unknown',
        is_symmetric    INTEGER DEFAULT 0,
        asymmetry_param REAL    DEFAULT 0.0,
        reference       TEXT    DEFAULT '',
        notes           TEXT    DEFAULT '',
        submitted_by    TEXT    DEFAULT 'anonymous',
        timestamp       TEXT    NOT NULL
    );
    """

    def __init__(self, db_path: str = DEFAULT_DB):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(self.CREATE_TABLE)
            conn.commit()

    def submit(self, obs: Observation) -> int:
        """
        Submit a new observation to the database.

        Parameters
        ----------
        obs : Observation

        Returns
        -------
        int — row ID of the inserted record
        """
        row = asdict(obs)
        row['is_symmetric'] = int(row['is_symmetric'])

        with self._connect() as conn:
            cur = conn.execute("""
                INSERT INTO observations
                (system_name, obs_type, delta_t_s, delta_t_err_s,
                 d1_m, d2_m, metric_type, is_symmetric, asymmetry_param,
                 reference, notes, submitted_by, timestamp)
                VALUES
                (:system_name, :obs_type, :delta_t_s, :delta_t_err_s,
                 :d1_m, :d2_m, :metric_type, :is_symmetric, :asymmetry_param,
                 :reference, :notes, :submitted_by, :timestamp)
            """, row)
            conn.commit()
            return cur.lastrowid

    def query(
        self,
        obs_type    : Optional[str] = None,
        system_name : Optional[str] = None,
        symmetric   : Optional[bool] = None,
        limit       : int = 100
    ) -> List[dict]:
        """
        Query observations from the database.

        Parameters
        ----------
        obs_type    : filter by type ('lensing'|'shapiro'|'computed'|'custom')
        system_name : filter by system name
        symmetric   : filter symmetric/asymmetric cases
        limit       : max rows to return

        Returns
        -------
        List[dict] — matching observation records
        """
        conditions = []
        params     = []

        if obs_type:
            conditions.append("obs_type = ?")
            params.append(obs_type)
        if system_name:
            conditions.append("system_name LIKE ?")
            params.append(f"%{system_name}%")
        if symmetric is not None:
            conditions.append("is_symmetric = ?")
            params.append(int(symmetric))

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                f"SELECT * FROM observations {where} "
                f"ORDER BY timestamp DESC LIMIT ?",
                params + [limit]
            )
            return [dict(row) for row in cur.fetchall()]

    def count(self) -> dict:
        """Return observation counts by type."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT obs_type, COUNT(*) as n FROM observations GROUP BY obs_type"
            )
            return {row[0]: row[1] for row in cur.fetchall()}

    def export_json(self, path: str):
        """Export all observations to a JSON file."""
        all_obs = self.query(limit=100000)
        with open(path, 'w') as f:
            json.dump(all_obs, f, indent=2)
        print(f"Exported {len(all_obs)} observations to {path}")

    def load_known_systems(self):
        """Pre-populate database with known systems from known_systems.py."""
        from .known_systems import KNOWN_SYSTEMS
        count = 0
        for sys in KNOWN_SYSTEMS:
            if sys['delta_t_obs'] is None:
                continue
            obs = Observation(
                system_name    = sys['name'],
                obs_type       = sys['type'],
                delta_t_s      = sys['delta_t_obs'],
                delta_t_err_s  = sys['delta_t_err'] or 0.0,
                reference      = sys['reference'],
                notes          = sys['description'],
                submitted_by   = 'NGO_preloaded'
            )
            self.submit(obs)
            count += 1
        print(f"Loaded {count} known systems into database.")

    def summary(self) -> str:
        counts = self.count()
        total  = sum(counts.values())
        lines  = [
            "=" * 50,
            "  NGO Observation Database  |  Chirag Rathi",
            "=" * 50,
            f"  Total observations : {total}",
        ]
        for t, n in counts.items():
            lines.append(f"    {t:20s} : {n}")
        lines.append(f"  Database file : {self.db_path}")
        lines.append("=" * 50)
        return "\n".join(lines)
