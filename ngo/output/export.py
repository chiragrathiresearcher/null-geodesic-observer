# ============================================================
# NGO — Null Geodesic Observer
# Module : ngo/output/export.py
# Author : Chirag Rathi
# Purpose: Export results to CSV, JSON, LaTeX table
# ============================================================
"""
export.py
---------
Export NGO results in formats suitable for:
  - Data analysis (CSV, JSON)
  - Paper submission (LaTeX table)
  - Archive/sharing (JSON with full metadata)
"""

import json
import csv
import os
import numpy as np
from datetime import datetime
from ..core.delta_t import DeltaTResult


def export_delta_t_csv(result: DeltaTResult, path: str):
    """
    Export a DeltaTResult to CSV.

    Parameters
    ----------
    result : DeltaTResult
    path   : str — output file path (.csv)
    """
    rows = [
        ['Parameter', 'Value', 'Unit'],
        ['delta_t',          result.delta_t,              's'],
        ['delta_t_days',     result.delta_t_days,         'days'],
        ['delta_t_us',       result.delta_t_microseconds, 'μs'],
        ['t1',               result.t1,                   's'],
        ['t2',               result.t2,                   's'],
        ['L1',               result.L1,                   'm'],
        ['L2',               result.L2,                   'm'],
        ['d1_coord',         result.d1,                   'm'],
        ['d2_coord',         result.d2,                   'm'],
        ['proper_d1',        result.proper_d1,            'm'],
        ['proper_d2',        result.proper_d2,            'm'],
        ['is_symmetric',     result.is_symmetric,         'bool'],
        ['asymmetry_param',  result.asymmetry_param,      'dimensionless'],
        ['method',           result.method,               ''],
        ['notes',            result.notes,                ''],
        ['generated_by',     'NGO | Chirag Rathi',        ''],
        ['timestamp',        datetime.utcnow().isoformat(),''],
    ]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Exported CSV → {path}")


def export_delta_t_json(result: DeltaTResult, path: str):
    """
    Export a DeltaTResult to JSON with full metadata.

    Parameters
    ----------
    result : DeltaTResult
    path   : str — output file path (.json)
    """
    data = {
        'metadata': {
            'software'  : 'NGO — Null Geodesic Observer',
            'author'    : 'Chirag Rathi',
            'version'   : '0.1.0',
            'timestamp' : datetime.utcnow().isoformat(),
            'paper'     : (
                'Null Geodesic Path Integrals and Observational Signatures '
                'of Spacetime Curvature: From Shapiro Delay to Cosmological Lensing'
            ),
        },
        'result': {
            'delta_t_s'          : result.delta_t,
            'delta_t_days'       : result.delta_t_days,
            'delta_t_years'      : result.delta_t_years,
            'delta_t_us'         : result.delta_t_microseconds,
            't1_s'               : result.t1,
            't2_s'               : result.t2,
            'L1_m'               : result.L1,
            'L2_m'               : result.L2,
            'd1_coord_m'         : result.d1,
            'd2_coord_m'         : result.d2,
            'proper_d1_m'        : result.proper_d1,
            'proper_d2_m'        : result.proper_d2,
            'is_symmetric'       : result.is_symmetric,
            'asymmetry_parameter': result.asymmetry_param,
            'method'             : result.method,
            'notes'              : result.notes,
        },
        'geodesic_1': {
            'n_steps'          : result.geodesic_1.n_steps,
            'success'          : result.geodesic_1.success,
            'max_null_violation': result.geodesic_1.max_null_violation(),
        },
        'geodesic_2': {
            'n_steps'          : result.geodesic_2.n_steps,
            'success'          : result.geodesic_2.success,
            'max_null_violation': result.geodesic_2.max_null_violation(),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Exported JSON → {path}")


def export_latex_table(results: list, path: str, caption: str = None):
    """
    Export a list of DeltaTResult or observation dicts to a
    LaTeX table — ready for direct inclusion in the paper.

    Parameters
    ----------
    results : list of dict or DeltaTResult
    path    : str — output .tex file path
    caption : str — table caption
    """
    if caption is None:
        caption = (
            "Light travel time asymmetry measurements. "
            "Columns: system name, observed $\\Delta t$, "
            "NGO predicted $\\Delta t$, fractional residual, "
            "asymmetry parameter $\\mathcal{A}$."
        )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{tab:delta_t_results}",
        r"\begin{tabular}{lrrrc}",
        r"\hline\hline",
        r"System & $\Delta t_\mathrm{obs}$ & "
        r"$\Delta t_\mathrm{pred}$ & "
        r"Residual & $\mathcal{A}$ \\",
        r" & [days] & [days] & [\%] & \\",
        r"\hline",
    ]

    for r in results:
        if isinstance(r, DeltaTResult):
            name   = "NGO computed"
            obs    = "—"
            pred   = f"{r.delta_t_days:.2f}"
            resid  = "—"
            asym   = f"{r.asymmetry_param:.4f}"
        else:
            name   = r.get('system', '—')
            obs_s  = r.get('observed_s')
            pred_s = r.get('predicted_s')
            obs    = f"{obs_s/86400:.2f}"  if obs_s  is not None else "—"
            pred   = f"{pred_s/86400:.4f}" if pred_s is not None else "—"
            frac   = r.get('frac_residual')
            resid  = f"{frac*100:.2f}"     if frac   is not None else "—"
            asym   = "—"

        lines.append(f"{name} & {obs} & {pred} & {resid} & {asym} \\\\")

    lines += [
        r"\hline\hline",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item Generated by NGO (Null Geodesic Observer) — Chirag Rathi (2026)",
        r"\end{tablenotes}",
        r"\end{table}",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Exported LaTeX table → {path}")


def export_geodesic_path_csv(sol, path: str, label: str = "geodesic"):
    """
    Export a GeodesicSolution path to CSV for external plotting.

    Columns: lambda, t, x1, x2, x3, k0, k1, k2, k3, null_violation
    """
    N = sol.x.shape[1]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'lambda', 't', 'x1', 'x2', 'x3',
            'k0', 'k1', 'k2', 'k3', 'null_violation',
            'label', 'author'
        ])
        for i in range(N):
            nv = sol.null_violation[i] if len(sol.null_violation) > i else 0.0
            writer.writerow([
                sol.lam[i],
                sol.x[0, i], sol.x[1, i], sol.x[2, i], sol.x[3, i],
                sol.k[0, i], sol.k[1, i], sol.k[2, i], sol.k[3, i],
                nv, label, 'Chirag Rathi'
            ])
    print(f"Exported geodesic path → {path}")
