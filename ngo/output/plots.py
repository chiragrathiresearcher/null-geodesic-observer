# ============================================================
# NGO — Null Geodesic Observer
# Module : ngo/output/plots.py
# Author : Chirag Rathi
# Purpose: Matplotlib plots for paper figures and analysis
# ============================================================
"""
plots.py
--------
Generates all static plots using Matplotlib.

Figures produced:
  1. plot_geodesic_pair()     — Case I vs Case II (flat vs curved)
  2. plot_delta_t_vs_b()      — Δt as function of impact parameter
  3. plot_regime_scan()       — weak vs strong field comparison
  4. plot_known_systems()     — observed Δt for all known systems
  5. plot_asymmetry_map()     — asymmetry parameter across configuration space
  6. plot_refractive_index()  — n(x) along two geodesics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# NGO style
NGO_STYLE = {
    'axes.facecolor'   : '#0d1117',
    'figure.facecolor' : '#0d1117',
    'axes.edgecolor'   : '#30363d',
    'axes.labelcolor'  : '#e6edf3',
    'xtick.color'      : '#8b949e',
    'ytick.color'      : '#8b949e',
    'text.color'       : '#e6edf3',
    'grid.color'       : '#21262d',
    'grid.linestyle'   : '--',
    'grid.alpha'       : 0.5,
    'font.family'      : 'serif',
    'font.size'        : 11,
}

COL1 = '#58a6ff'   # blue  — geodesic 1
COL2 = '#f78166'   # red   — geodesic 2
COL3 = '#3fb950'   # green — flat spacetime
COLA = '#d2a8ff'   # purple — analytic
COLW = '#ffa657'   # orange — weak field


def _apply_style():
    plt.rcParams.update(NGO_STYLE)


def _watermark(ax):
    ax.text(
        0.99, 0.01, 'NGO | Chirag Rathi',
        transform=ax.transAxes,
        fontsize=7, color='#484f58',
        ha='right', va='bottom', style='italic'
    )


def plot_geodesic_pair(
    sol1, sol2,
    title   : str = "Null Geodesic Pair",
    savepath: str = None
):
    """
    Plot two null geodesic paths in the x-y plane.
    Reproduces the Case I / Case II diagrams from Chirag's notes.

    Parameters
    ----------
    sol1, sol2 : GeodesicSolution — from GeodesicSolver.solve()
    title      : str
    savepath   : str or None — if given, saves figure to path
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    x1 = sol1.x[1, :]   # spatial x-coordinate
    y1 = sol1.x[2, :]   # spatial y-coordinate
    x2 = sol2.x[1, :]
    y2 = sol2.x[2, :]

    ax.plot(x1, y1, color=COL1, lw=2, label='Geodesic γ₁ (Source 1)', zorder=3)
    ax.plot(x2, y2, color=COL2, lw=2, label='Geodesic γ₂ (Source 2)', zorder=3)

    # mark start and end points
    ax.scatter(x1[0],  y1[0],  color=COL1, s=80, zorder=5, marker='o')
    ax.scatter(x2[0],  y2[0],  color=COL2, s=80, zorder=5, marker='o')
    ax.scatter(x1[-1], y1[-1], color='white', s=120, zorder=5,
               marker='*', label='Observer O')

    ax.set_xlabel('x¹  [m]')
    ax.set_ylabel('x²  [m]')
    ax.set_title(title, pad=12)
    ax.legend(framealpha=0.2, loc='upper left')
    ax.grid(True)
    _watermark(ax)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_delta_t_vs_b(
    b_values   : np.ndarray,
    delta_t    : np.ndarray,
    delta_t_an : np.ndarray = None,
    M          : float = None,
    savepath   : str = None
):
    """
    Plot Δt as a function of impact parameter b.
    Shows transition from weak-field to strong-field regime.

    Parameters
    ----------
    b_values   : np.ndarray — impact parameters [m]
    delta_t    : np.ndarray — exact Δt values [s]
    delta_t_an : np.ndarray — analytic weak-field Δt [s] (optional)
    M          : float — lens mass [kg] (for r_s annotation)
    savepath   : str or None
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(b_values, delta_t * 1e6, color=COL1, lw=2,
            label='Exact (numerical)', zorder=3)

    if delta_t_an is not None:
        ax.plot(b_values, delta_t_an * 1e6, color=COLA, lw=1.5,
                ls='--', label='Weak-field analytic', zorder=2)

    if M is not None:
        rs  = 2 * 6.674e-11 * M / (2.998e8)**2
        rph = 1.5 * rs
        ax.axvline(rs,  color='#ff7b72', lw=1, ls=':', alpha=0.8,
                   label=f'Schwarzschild radius r_s')
        ax.axvline(rph, color=COLW, lw=1, ls=':', alpha=0.8,
                   label=f'Photon sphere 1.5 r_s')

    ax.set_xscale('log')
    ax.set_xlabel('Impact parameter b  [m]')
    ax.set_ylabel('Δt  [μs]')
    ax.set_title('Shapiro Delay vs Impact Parameter\n'
                 'Exact (NGO) vs Weak-Field Analytic')
    ax.legend(framealpha=0.2)
    ax.grid(True)
    _watermark(ax)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig, ax


def plot_known_systems(savepath: str = None):
    """
    Bar chart of observed Δt for all known lensing/Shapiro systems.
    Figure for paper Section 4.
    """
    from ..database.known_systems import KNOWN_SYSTEMS

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={'width_ratios': [2, 1]})

    # ── left: lensing systems (days) ─────────────────────────
    ax = axes[0]
    lens_sys = [s for s in KNOWN_SYSTEMS
                if s['type'] == 'lensing' and s['delta_t_obs'] is not None]

    names  = [s['common_name'] for s in lens_sys]
    dt_d   = [s['delta_t_obs'] / 86400 for s in lens_sys]
    err_d  = [s['delta_t_err'] / 86400 for s in lens_sys]
    colors = [COL1, COL2, COL3, COLA][:len(lens_sys)]

    bars = ax.barh(names, dt_d, xerr=err_d, color=colors,
                   alpha=0.85, capsize=5, edgecolor='#30363d')
    ax.set_xlabel('Δt  [days]')
    ax.set_title('Gravitational Lensing Time Delays\n(Known Systems)')
    ax.grid(True, axis='x')
    _watermark(ax)

    # ── right: Shapiro (μs) ──────────────────────────────────
    ax2 = axes[1]
    shap_sys = [s for s in KNOWN_SYSTEMS
                if s['type'] == 'shapiro' and s['delta_t_obs'] is not None]

    names2 = [s['common_name'] for s in shap_sys]
    dt_us  = [s['delta_t_obs'] * 1e6 for s in shap_sys]
    err_us = [(s['delta_t_err'] or 0) * 1e6 for s in shap_sys]

    ax2.barh(names2, dt_us, xerr=err_us, color=COLW,
             alpha=0.85, capsize=5, edgecolor='#30363d')
    ax2.set_xlabel('Δt  [μs]')
    ax2.set_title('Shapiro Delay\n(Solar System)')
    ax2.grid(True, axis='x')
    _watermark(ax2)

    plt.suptitle('Observed Light Travel Time Asymmetry — Chirag Rathi (NGO)',
                 fontsize=13, y=1.02)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig, axes


def plot_refractive_index(
    sol1, sol2,
    metric,
    savepath: str = None
):
    """
    Plot the effective refractive index n(x) along two geodesics.
    Shows physically why Δt ≠ 0 — different n(x) histories.
    """
    from ..core.path_integral import effective_refractive_index

    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    for sol, ax, col, label in [
        (sol1, axes[0], COL1, 'Geodesic γ₁'),
        (sol2, axes[1], COL2, 'Geodesic γ₂'),
    ]:
        N = sol.x.shape[1]
        n_vals = np.zeros(N)
        for i in range(N):
            try:
                n_vals[i] = effective_refractive_index(metric, sol.x[:, i])
            except Exception:
                n_vals[i] = 1.0

        sigma = np.linspace(0, 1, N)
        ax.plot(sigma, n_vals, color=col, lw=2, label=label)
        ax.axhline(1.0, color='#484f58', lw=1, ls='--', label='Flat n=1')
        ax.fill_between(sigma, 1.0, n_vals,
                        where=(n_vals > 1.0), alpha=0.2, color=col)
        ax.set_ylabel('n(x) = √(−g₀₀)')
        ax.legend(framealpha=0.2)
        ax.grid(True)
        _watermark(ax)

    axes[1].set_xlabel('Normalised affine parameter λ/λ_max')
    axes[0].set_title('Effective Refractive Index Along Each Null Geodesic\n'
                      'Area difference → Δt ≠ 0')
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig, axes


def plot_null_violation(sol1, sol2, savepath: str = None):
    """
    Plot the null condition violation g_μν k^μ k^ν along both geodesics.
    Quality/accuracy diagnostic — should be ~0 throughout.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 4))

    for sol, col, label in [
        (sol1, COL1, 'Geodesic γ₁'),
        (sol2, COL2, 'Geodesic γ₂'),
    ]:
        if len(sol.null_violation) > 0:
            ax.semilogy(sol.lam, np.abs(sol.null_violation) + 1e-30,
                        color=col, lw=1.5, label=label)

    ax.set_xlabel('Affine parameter λ')
    ax.set_ylabel('|g_μν k^μ k^ν|  (null violation)')
    ax.set_title('Null Condition Accuracy Along Geodesics\n'
                 '(Quality Diagnostic — should remain ≪ 1)')
    ax.legend(framealpha=0.2)
    ax.grid(True)
    _watermark(ax)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig, ax
