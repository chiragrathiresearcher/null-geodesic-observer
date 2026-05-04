# ============================================================
# NGO — Null Geodesic Observer
# Module : ngo/database/known_systems.py
# Author : Chirag Rathi
# Purpose: Pre-loaded observational data for known systems
# ============================================================
"""
known_systems.py
----------------
Pre-loaded database of systems with measured light travel
time asymmetry (gravitational lensing time delays, Shapiro delay).

Each system is a dict with:
    name         : str   — system identifier
    type         : str   — 'lensing' | 'shapiro' | 'binary_pulsar'
    delta_t_obs  : float — observed Δt in seconds
    delta_t_err  : float — measurement uncertainty in seconds
    description  : str   — physical description
    reference    : str   — citation
    parameters   : dict  — physical parameters of the system

Sources:
  Shapiro (1964), Refsdal (1964), Bertotti et al. (2003),
  Kundic et al. (1997), Biggs et al. (1999), Kelly et al. (2016)
"""

KNOWN_SYSTEMS = [

    # ── GRAVITATIONAL LENSING SYSTEMS ──────────────────────
    {
        "name"        : "Q0957+561",
        "type"        : "lensing",
        "common_name" : "Twin Quasar",
        "delta_t_obs" : 417.0 * 86400.0,    # 417 days in seconds
        "delta_t_err" :   3.0 * 86400.0,    # ±3 days
        "description" : (
            "First discovered gravitational lens system. "
            "Quasar at z=1.41 lensed by galaxy at z=0.36. "
            "Two images A and B separated by 6 arcseconds. "
            "Image B leads image A by ~417 days."
        ),
        "reference"   : "Kundic et al. (1997), ApJ 482, 75",
        "parameters"  : {
            "z_source"      : 1.41,
            "z_lens"        : 0.36,
            "image_sep_arcsec": 6.17,
            "delta_t_days"  : 417.09,
            "H0_implied"    : 63.1,     # km/s/Mpc
        }
    },

    {
        "name"        : "B0218+357",
        "type"        : "lensing",
        "common_name" : "Lensed Blazar B0218",
        "delta_t_obs" : 10.5 * 86400.0,
        "delta_t_err" :  0.4 * 86400.0,
        "description" : (
            "Compact radio source lensed by a spiral galaxy. "
            "Smallest image separation of any known lens (335 mas). "
            "Δt ≈ 10.5 days."
        ),
        "reference"   : "Biggs et al. (1999), MNRAS 304, 349",
        "parameters"  : {
            "z_source"         : 0.944,
            "z_lens"           : 0.685,
            "image_sep_mas"    : 335.0,
            "delta_t_days"     : 10.5,
        }
    },

    {
        "name"        : "HE0435-1223",
        "type"        : "lensing",
        "common_name" : "HE0435 Quad Lens",
        "delta_t_obs" : 14.4 * 86400.0,
        "delta_t_err" :  0.8 * 86400.0,
        "description" : (
            "Quadruply lensed quasar. Time delay between "
            "images A-D = 14.4 days. Used for H0 measurement."
        ),
        "reference"   : "Bonvin et al. (2017), MNRAS 465, 4914",
        "parameters"  : {
            "z_source"     : 1.693,
            "z_lens"       : 0.454,
            "n_images"     : 4,
            "delta_t_days" : 14.4,
        }
    },

    {
        "name"        : "SN_Refsdal",
        "type"        : "lensing",
        "common_name" : "Supernova Refsdal",
        "delta_t_obs" : 376.0 * 86400.0,
        "delta_t_err" :  24.0 * 86400.0,
        "description" : (
            "First multiply imaged supernova. Lensed by galaxy cluster "
            "MACS J1149.6+2223 at z=0.544. Source SN at z=1.49. "
            "Named in honour of Sjur Refsdal who predicted this effect."
        ),
        "reference"   : "Kelly et al. (2016), ApJ 819, L8",
        "parameters"  : {
            "z_source"     : 1.49,
            "z_lens"       : 0.544,
            "delta_t_days" : 376.0,
        }
    },

    # ── SHAPIRO DELAY SYSTEMS ───────────────────────────────
    {
        "name"        : "Cassini_2003",
        "type"        : "shapiro",
        "common_name" : "Cassini Spacecraft Solar Conjunction",
        "delta_t_obs" : 246.0e-6,     # 246 microseconds
        "delta_t_err" :   0.05e-6,    # ±0.05 μs
        "description" : (
            "Most precise Shapiro delay measurement to date. "
            "Radio signals from Cassini passed near the Sun during "
            "superior conjunction. Tested GR to 20 ppm. "
            "Measures the PPN parameter γ = 1 + (2.1 ± 2.3) × 10⁻⁵."
        ),
        "reference"   : "Bertotti, Iess & Tortora (2003), Nature 425, 374",
        "parameters"  : {
            "M_lens"        : 1.989e30,   # solar mass [kg]
            "b_min_solar_r" : 1.6,        # closest approach in solar radii
            "PPN_gamma"     : 1.000021,
            "uncertainty_ppm": 20,
        }
    },

    {
        "name"        : "Hulse_Taylor",
        "type"        : "binary_pulsar",
        "common_name" : "Hulse-Taylor Binary Pulsar PSR B1913+16",
        "delta_t_obs" : None,         # time-varying; orbital period dependent
        "delta_t_err" : None,
        "description" : (
            "First binary pulsar. Provides strong-field test of GR "
            "through orbital decay due to gravitational wave emission. "
            "Shapiro delay component measurable in pulse timing. "
            "Nobel Prize 1993."
        ),
        "reference"   : "Hulse & Taylor (1975), ApJ 195, L51",
        "parameters"  : {
            "P_orbital_hours" : 7.75,
            "eccentricity"    : 0.617,
            "M_pulsar"        : 1.44,    # solar masses
            "M_companion"     : 1.39,    # solar masses
        }
    },

]


def get_system(name: str) -> dict:
    """
    Retrieve a known system by name.

    Parameters
    ----------
    name : str — system name (e.g. 'Q0957+561', 'Cassini_2003')

    Returns
    -------
    dict — system data
    """
    for sys in KNOWN_SYSTEMS:
        if sys['name'].lower() == name.lower():
            return sys
    available = [s['name'] for s in KNOWN_SYSTEMS]
    raise KeyError(f"System '{name}' not found. Available: {available}")


def list_systems(system_type: str = None) -> list:
    """
    List all known systems, optionally filtered by type.

    Parameters
    ----------
    system_type : str or None — 'lensing' | 'shapiro' | 'binary_pulsar'

    Returns
    -------
    list of system names
    """
    if system_type is None:
        return [s['name'] for s in KNOWN_SYSTEMS]
    return [s['name'] for s in KNOWN_SYSTEMS if s['type'] == system_type]


def get_all_delta_t() -> dict:
    """
    Return all measured Δt values as a simple dict.

    Returns
    -------
    dict: {name: delta_t_seconds}
    """
    return {
        s['name']: s['delta_t_obs']
        for s in KNOWN_SYSTEMS
        if s['delta_t_obs'] is not None
    }
