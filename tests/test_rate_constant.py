import sys
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

from pykmc.rate_constant import (
    EV_PER_A2_AMU_TO_RAD2_PER_S2,
    compute_rate_amsel_vtst,
    compute_rate_amsel_vtst_details,
    finite_difference_hessian_from_forces,
    mass_weighted_hessian_eigenvalues,
    vineyard_event_prefactors_from_forces,
    vineyard_projected_event_prefactors_from_forces,
    vineyard_prefactor_from_eigenvalues,
    vineyard_prefactor_from_hessians,
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


def test_finite_difference_hessian_uses_negative_force_derivative():
    hessian = np.diag([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    positions = np.zeros((3, 3), dtype=float)
    active_indices = [0, 2]

    def harmonic_forces(displaced):
        active_flat = displaced[active_indices].reshape(-1)
        forces = np.zeros_like(displaced)
        forces[active_indices] = (-hessian @ active_flat).reshape(-1, 3)
        return forces

    got = finite_difference_hessian_from_forces(
        harmonic_forces,
        positions,
        active_indices,
        step_A=1.0e-4,
    )

    np.testing.assert_allclose(got, hessian, rtol=1.0e-10, atol=1.0e-10)


def test_mass_weighted_hessian_eigenvalues_apply_atomic_masses():
    hessian = np.diag([4.0, 9.0, 16.0])

    got = mass_weighted_hessian_eigenvalues(hessian, masses_amu=[2.0])

    expected = np.array([2.0, 4.5, 8.0]) * EV_PER_A2_AMU_TO_RAD2_PER_S2
    np.testing.assert_allclose(got, expected)


def test_vineyard_prefactor_from_hessians_mass_weights_before_ratio():
    minimum_hessian = np.diag([4.0, 9.0, 16.0])
    saddle_hessian = np.diag([-1.0, 4.0, 9.0])

    prefactor = vineyard_prefactor_from_hessians(
        minimum_hessian,
        saddle_hessian,
        masses_amu=[1.0],
    )

    expected = np.sqrt(16.0 * EV_PER_A2_AMU_TO_RAD2_PER_S2) / (2.0 * np.pi)
    assert prefactor == pytest.approx(expected)


def test_vineyard_event_prefactors_from_forces_returns_forward_and_backward():
    min1_hessian = np.diag([4.0, 9.0, 16.0])
    min2_hessian = np.diag([25.0, 9.0, 16.0])
    saddle_hessian = np.diag([-1.0, 9.0, 16.0])
    min1_positions = np.array([[0.0, 0.0, 0.0]], dtype=float)
    min2_positions = np.array([[10.0, 0.0, 0.0]], dtype=float)
    saddle_positions = np.array([[20.0, 0.0, 0.0]], dtype=float)

    def forces(displaced):
        if np.linalg.norm(displaced - min1_positions) < 1.0:
            return (-(min1_hessian @ (displaced - min1_positions).reshape(-1))).reshape(
                -1, 3
            )
        if np.linalg.norm(displaced - min2_positions) < 1.0:
            return (-(min2_hessian @ (displaced - min2_positions).reshape(-1))).reshape(
                -1, 3
            )
        return (-(saddle_hessian @ (displaced - saddle_positions).reshape(-1))).reshape(
            -1, 3
        )

    prefactors = vineyard_event_prefactors_from_forces(
        forces,
        min1_positions,
        saddle_positions,
        min2_positions,
        active_indices=[0],
        masses_amu=[1.0],
        step_A=1.0e-4,
    )

    conv = EV_PER_A2_AMU_TO_RAD2_PER_S2
    assert prefactors.forward_prefactor_inv_s == pytest.approx(
        np.sqrt(4.0 * conv) / (2.0 * np.pi)
    )
    assert prefactors.backward_prefactor_inv_s == pytest.approx(
        np.sqrt(25.0 * conv) / (2.0 * np.pi)
    )
    assert prefactors.barrier_omega_rad_per_s == pytest.approx(np.sqrt(conv))
    assert prefactors.saddle_freq_invcm == pytest.approx(
        np.sqrt(conv) / (2.0 * np.pi * 2.99792458e10)
    )


def test_vineyard_event_prefactors_report_hessian_stages():
    min1_hessian = np.diag([4.0, 9.0, 16.0])
    min2_hessian = np.diag([25.0, 9.0, 16.0])
    saddle_hessian = np.diag([-1.0, 9.0, 16.0])
    min1_positions = np.array([[0.0, 0.0, 0.0]], dtype=float)
    min2_positions = np.array([[10.0, 0.0, 0.0]], dtype=float)
    saddle_positions = np.array([[20.0, 0.0, 0.0]], dtype=float)

    def forces(displaced):
        if np.linalg.norm(displaced - min1_positions) < 1.0:
            return (-(min1_hessian @ (displaced - min1_positions).reshape(-1))).reshape(
                -1, 3
            )
        if np.linalg.norm(displaced - min2_positions) < 1.0:
            return (-(min2_hessian @ (displaced - min2_positions).reshape(-1))).reshape(
                -1, 3
            )
        return (-(saddle_hessian @ (displaced - saddle_positions).reshape(-1))).reshape(
            -1, 3
        )

    stages = []

    vineyard_event_prefactors_from_forces(
        forces,
        min1_positions,
        saddle_positions,
        min2_positions,
        active_indices=[0],
        masses_amu=[1.0],
        step_A=1.0e-4,
        progress_callback=lambda stage, status, elapsed_s=None: stages.append(
            (stage, status, elapsed_s)
        ),
    )

    assert [(stage, status) for stage, status, _elapsed in stages] == [
        ("minimum_forward", "start"),
        ("minimum_forward", "complete"),
        ("minimum_backward", "start"),
        ("minimum_backward", "complete"),
        ("saddle", "start"),
        ("saddle", "complete"),
    ]
    assert all(
        elapsed_s is None or elapsed_s >= 0.0
        for _stage, _status, elapsed_s in stages
    )


def test_vineyard_projected_prefactors_follow_reaction_coordinate_curvature():
    min1_hessian = np.diag([4.0, 9.0, 16.0])
    min2_hessian = np.diag([25.0, 9.0, 16.0])
    saddle_hessian = np.diag([-1.0, 9.0, 16.0])
    min1_positions = np.array([[0.0, 0.0, 0.0]], dtype=float)
    saddle_positions = np.array([[1.0, 0.0, 0.0]], dtype=float)
    min2_positions = np.array([[2.0, 0.0, 0.0]], dtype=float)

    def forces(displaced):
        if np.linalg.norm(displaced - min1_positions) < 0.5:
            return (-(min1_hessian @ (displaced - min1_positions).reshape(-1))).reshape(
                -1, 3
            )
        if np.linalg.norm(displaced - min2_positions) < 0.5:
            return (-(min2_hessian @ (displaced - min2_positions).reshape(-1))).reshape(
                -1, 3
            )
        return (-(saddle_hessian @ (displaced - saddle_positions).reshape(-1))).reshape(
            -1, 3
        )

    prefactors = vineyard_projected_event_prefactors_from_forces(
        forces,
        min1_positions,
        saddle_positions,
        min2_positions,
        active_indices=[0],
        masses_amu=[1.0],
        step_A=1.0e-4,
    )

    conv = EV_PER_A2_AMU_TO_RAD2_PER_S2
    assert prefactors.forward_prefactor_inv_s == pytest.approx(
        np.sqrt(4.0 * conv) / (2.0 * np.pi)
    )
    assert prefactors.backward_prefactor_inv_s == pytest.approx(
        np.sqrt(25.0 * conv) / (2.0 * np.pi)
    )
    assert prefactors.barrier_omega_rad_per_s == pytest.approx(np.sqrt(conv))
    assert prefactors.saddle_freq_invcm == pytest.approx(
        np.sqrt(conv) / (2.0 * np.pi * 2.99792458e10)
    )
