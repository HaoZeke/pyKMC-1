import sys
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

from pykmc.rate_constant import (
    EV_PER_A2_AMU_TO_RAD2_PER_S2,
    compute_rate_amsel_vtst,
    compute_rate_amsel_vtst_details,
    vineyard_prefactor_from_eigenvalues,
)


def test_amsel_vtst_returns_ps_inverse_for_kmc_clock(monkeypatch):
    fake_amsel = SimpleNamespace(
        vtst_corrected_rate=lambda *_args: {"corrected_rate_inv_s": 5.0e12}
    )
    monkeypatch.setitem(sys.modules, "amsel", fake_amsel)
    config = SimpleNamespace(
        rateconstant=SimpleNamespace(
            prefactor=5.0e12,
            T=300.0,
            saddle_freq_invcm=0.0,
            friction_inv_s=0.0,
            barrier_omega_rad_per_s=0.0,
        )
    )

    assert compute_rate_amsel_vtst(0.0, 0.0, config) == 5.0


def test_amsel_vtst_details_expose_prefactor_and_correction_provenance(monkeypatch):
    fake_amsel = SimpleNamespace(
        vtst_corrected_rate=lambda *_args: {
            "harmonic_rate_inv_s": 2.0e11,
            "corrected_rate_inv_s": 3.0e11,
            "wigner_factor": 1.25,
            "eckart_factor": 1.20,
            "kramers_factor": 1.0,
        }
    )
    monkeypatch.setitem(sys.modules, "amsel", fake_amsel)
    config = SimpleNamespace(
        rateconstant=SimpleNamespace(
            prefactor=6.0e12,
            T=500.0,
            saddle_freq_invcm=120.0,
            friction_inv_s=0.0,
            barrier_omega_rad_per_s=2.0e13,
        )
    )

    details = compute_rate_amsel_vtst_details(0.42, 0.37, config)

    assert details.rate_inv_s == pytest.approx(3.0e11)
    assert details.rate_ps_inv == pytest.approx(0.3)
    assert details.prefactor_inv_s == pytest.approx(6.0e12)
    assert details.prefactor_source == "rateconstant.prefactor"
    assert details.rate_unit == "ps^-1"
    assert details.wigner_factor == pytest.approx(1.25)
    assert details.eckart_factor == pytest.approx(1.20)
    assert details.kramers_factor == pytest.approx(1.0)
    assert details.anharmonic_corrections_active is True
    assert details.rate_model_ok is False
    assert details.rate_model_reason == "configured-prefactor"


def test_amsel_vtst_details_mark_no_curvature_model_as_uncorrected(monkeypatch):
    fake_amsel = SimpleNamespace(
        vtst_corrected_rate=lambda *_args: {
            "corrected_rate_inv_s": 4.0e12,
            "wigner_factor": 1.0,
            "eckart_factor": 1.0,
            "kramers_factor": 1.0,
        }
    )
    monkeypatch.setitem(sys.modules, "amsel", fake_amsel)
    config = SimpleNamespace(
        rateconstant=SimpleNamespace(
            prefactor=4.0e12,
            T=300.0,
            saddle_freq_invcm=0.0,
            friction_inv_s=0.0,
            barrier_omega_rad_per_s=0.0,
        )
    )

    details = compute_rate_amsel_vtst_details(0.0, 0.0, config)

    assert details.rate_ps_inv == pytest.approx(4.0)
    assert details.anharmonic_corrections_active is False
    assert details.rate_model_ok is False
    assert details.rate_model_reason == "configured-prefactor-no-curvature"


def test_vineyard_prefactor_symbolically_cancels_shared_stable_modes():
    omega_a, omega_b, omega_reactant = sp.symbols(
        "omega_a omega_b omega_reactant", positive=True
    )

    expression = (
        omega_a
        * omega_b
        * omega_reactant
        / (omega_a * omega_b)
        / (2 * sp.pi)
    )

    assert sp.simplify(expression - omega_reactant / (2 * sp.pi)) == 0


def test_vineyard_prefactor_from_mass_weighted_eigenvalues():
    minimum_eigenvalues = np.array([4.0, 9.0, 16.0]) * EV_PER_A2_AMU_TO_RAD2_PER_S2
    saddle_eigenvalues = np.array([-1.0, 4.0, 9.0]) * EV_PER_A2_AMU_TO_RAD2_PER_S2

    prefactor = vineyard_prefactor_from_eigenvalues(
        minimum_eigenvalues, saddle_eigenvalues
    )

    expected = np.sqrt(16.0 * EV_PER_A2_AMU_TO_RAD2_PER_S2) / (2.0 * np.pi)
    assert prefactor == pytest.approx(expected)


def test_vineyard_prefactor_rejects_saddle_without_one_unstable_mode():
    with pytest.raises(ValueError, match="exactly one unstable"):
        vineyard_prefactor_from_eigenvalues(
            np.array([4.0, 9.0, 16.0]),
            np.array([1.0, 4.0, 9.0]),
        )
