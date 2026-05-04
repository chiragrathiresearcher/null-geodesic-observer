# ============================================================
# NGO — Null Geodesic Observer
# Tests  : tests/test_known_systems.py
# Author : Chirag Rathi
# ============================================================
"""
Tests for ngo.database.known_systems

Validates that all pre-loaded systems have:
  - Required fields
  - Physical plausibility of Δt values
  - Correct types
"""

import pytest
from ngo.database.known_systems import (
    KNOWN_SYSTEMS, get_system, list_systems, get_all_delta_t
)

REQUIRED_FIELDS = ['name', 'type', 'delta_t_obs', 'description', 'reference']
VALID_TYPES     = {'lensing', 'shapiro', 'binary_pulsar'}


class TestKnownSystemsStructure:

    def test_all_have_required_fields(self):
        for sys in KNOWN_SYSTEMS:
            for field in REQUIRED_FIELDS:
                assert field in sys, (
                    f"System '{sys.get('name','?')}' missing field '{field}'"
                )

    def test_all_types_valid(self):
        for sys in KNOWN_SYSTEMS:
            assert sys['type'] in VALID_TYPES, (
                f"System '{sys['name']}' has invalid type '{sys['type']}'"
            )

    def test_nonzero_delta_t(self):
        for sys in KNOWN_SYSTEMS:
            if sys['delta_t_obs'] is not None:
                assert sys['delta_t_obs'] > 0, (
                    f"System '{sys['name']}' has Δt ≤ 0"
                )

    def test_error_smaller_than_value(self):
        for sys in KNOWN_SYSTEMS:
            dt  = sys['delta_t_obs']
            err = sys['delta_t_err']
            if dt is not None and err is not None:
                assert err < dt, (
                    f"System '{sys['name']}': error {err} ≥ value {dt}"
                )


class TestKnownSystemsValues:

    def test_q0957_delta_t_days(self):
        sys = get_system('Q0957+561')
        dt_days = sys['delta_t_obs'] / 86400
        assert 400 < dt_days < 440, f"Q0957 Δt = {dt_days} days, expected ~417"

    def test_cassini_delta_t_microseconds(self):
        sys = get_system('Cassini_2003')
        dt_us = sys['delta_t_obs'] * 1e6
        assert 200 < dt_us < 300, f"Cassini Δt = {dt_us} μs, expected ~246"

    def test_b0218_delta_t_days(self):
        sys = get_system('B0218+357')
        dt_days = sys['delta_t_obs'] / 86400
        assert 9 < dt_days < 12, f"B0218 Δt = {dt_days} days, expected ~10.5"


class TestQueryFunctions:

    def test_get_system_found(self):
        sys = get_system('Q0957+561')
        assert sys['name'] == 'Q0957+561'

    def test_get_system_not_found(self):
        with pytest.raises(KeyError):
            get_system('NonExistentSystem_XYZ')

    def test_list_systems_all(self):
        names = list_systems()
        assert len(names) == len(KNOWN_SYSTEMS)
        assert 'Q0957+561' in names

    def test_list_systems_filtered(self):
        lensing = list_systems('lensing')
        for name in lensing:
            sys = get_system(name)
            assert sys['type'] == 'lensing'

    def test_get_all_delta_t_no_none(self):
        dts = get_all_delta_t()
        for name, dt in dts.items():
            assert dt is not None
            assert dt > 0
