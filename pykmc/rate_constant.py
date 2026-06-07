"""Module defining function used to compute the rate constant."""

from dataclasses import dataclass

from .config import PhysicalConstants, Config
import math as m
import numpy as np
import time

PS_PER_S = 1.0e12
EV_J = 1.602176634e-19
ANGSTROM_M = 1.0e-10
AMU_KG = 1.66053906660e-27
EV_PER_A2_AMU_TO_RAD2_PER_S2 = EV_J / (ANGSTROM_M**2 * AMU_KG)
SPEED_OF_LIGHT_CM_PER_S = 2.99792458e10


@dataclass(frozen=True)
class RateConstantDetails:
    """Rate value and provenance for a transition-rate calculation."""

    style: str
    rate_inv_s: float
    rate_ps_inv: float
    rate_unit: str
    harmonic_rate_inv_s: float
    prefactor_inv_s: float
    prefactor_source: str
    temperature_K: float
    dE_forward_ev: float
    dE_backward_ev: float
    saddle_freq_invcm: float
    friction_inv_s: float
    barrier_omega_rad_per_s: float
    wigner_factor: float
    eckart_factor: float
    kramers_factor: float
    anharmonic_corrections_active: bool
    rate_model_ok: bool
    rate_model_reason: str


@dataclass(frozen=True)
class VineyardEventPrefactors:
    """Forward and backward Vineyard prefactors for a transition event."""

    forward_prefactor_inv_s: float
    backward_prefactor_inv_s: float
    saddle_freq_invcm: float
    barrier_omega_rad_per_s: float


def vineyard_prefactor_from_eigenvalues(
    minimum_eigenvalues_rad2_per_s2,
    saddle_eigenvalues_rad2_per_s2,
    *,
    zero_tol: float = 0.0,
) -> float:
    r"""Return the harmonic Vineyard prefactor in ``s^-1``.

    The inputs are mass-weighted Hessian eigenvalues in angular-frequency
    squared units. The saddle spectrum must contain exactly one unstable
    mode. Positive stable modes enter as

    $$
    \nu = \frac{1}{2\pi}
          \frac{\prod_i \sqrt{\lambda_i^\mathrm{min}}}
               {\prod_j \sqrt{\lambda_j^\ddagger}}.
    $$
    """
    minimum = np.asarray(minimum_eigenvalues_rad2_per_s2, dtype=float).ravel()
    saddle = np.asarray(saddle_eigenvalues_rad2_per_s2, dtype=float).ravel()
    if minimum.size == 0 or saddle.size == 0:
        raise ValueError("Vineyard prefactor requires non-empty spectra")
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(saddle)):
        raise ValueError("Vineyard prefactor requires finite eigenvalues")
    tol = float(zero_tol)
    minimum_positive = minimum[minimum > tol]
    saddle_positive = saddle[saddle > tol]
    saddle_negative = saddle[saddle < -tol]
    if saddle_negative.size != 1:
        raise ValueError("Vineyard saddle spectrum must have exactly one unstable mode")
    if minimum_positive.size != saddle_positive.size + 1:
        raise ValueError(
            "Vineyard spectra must have one more minimum stable mode than saddle stable modes"
        )
    if minimum_positive.size == 0:
        raise ValueError("Vineyard prefactor requires stable minimum modes")
    log_prefactor = (
        0.5 * float(np.sum(np.log(minimum_positive)))
        - 0.5 * float(np.sum(np.log(saddle_positive)))
        - m.log(2.0 * m.pi)
    )
    return float(m.exp(log_prefactor))


def finite_difference_hessian_from_forces(
    force_fn,
    positions,
    active_indices,
    *,
    step_A: float = 1.0e-3,
) -> np.ndarray:
    """Build an active-block Hessian in ``eV/A^2`` from force evaluations."""
    base_positions = np.asarray(positions, dtype=float)
    active = np.asarray(active_indices, dtype=int).ravel()
    if base_positions.ndim != 2 or base_positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    if active.size == 0:
        raise ValueError("active_indices must be non-empty")
    if np.any(active < 0) or np.any(active >= base_positions.shape[0]):
        raise ValueError("active_indices must be valid atom indices")
    step = float(step_A)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_A must be positive")
    n_dof = int(active.size * 3)
    hessian = np.zeros((n_dof, n_dof), dtype=float)
    for column, (atom_index, axis) in enumerate(
        (atom, xyz) for atom in active for xyz in range(3)
    ):
        plus = base_positions.copy()
        minus = base_positions.copy()
        plus[atom_index, axis] += step
        minus[atom_index, axis] -= step
        force_plus = np.asarray(force_fn(plus), dtype=float)
        force_minus = np.asarray(force_fn(minus), dtype=float)
        if (
            force_plus.shape != base_positions.shape
            or force_minus.shape != base_positions.shape
        ):
            raise ValueError("force_fn must return forces with shape (n_atoms, 3)")
        force_derivative = (
            force_plus[active].reshape(-1) - force_minus[active].reshape(-1)
        ) / (2.0 * step)
        hessian[:, column] = -force_derivative
    return 0.5 * (hessian + hessian.T)


def mass_weighted_hessian_eigenvalues(hessian_ev_per_A2, masses_amu) -> np.ndarray:
    """Return mass-weighted Hessian eigenvalues in ``rad^2/s^2``."""
    hessian = np.asarray(hessian_ev_per_A2, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).ravel()
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be a square matrix")
    if hessian.shape[0] != masses.size * 3:
        raise ValueError("hessian size must be 3 * len(masses_amu)")
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(masses)):
        raise ValueError("hessian and masses must be finite")
    if np.any(masses <= 0.0):
        raise ValueError("masses_amu must be positive")
    dof_masses = np.repeat(masses, 3)
    mass_weighted = hessian / np.sqrt(np.outer(dof_masses, dof_masses))
    mass_weighted = 0.5 * (mass_weighted + mass_weighted.T)
    return np.linalg.eigvalsh(mass_weighted) * EV_PER_A2_AMU_TO_RAD2_PER_S2


def vineyard_prefactor_from_hessians(
    minimum_hessian_ev_per_A2,
    saddle_hessian_ev_per_A2,
    masses_amu,
    *,
    zero_tol_rad2_per_s2: float = 0.0,
) -> float:
    """Return the Vineyard prefactor from minimum and saddle Hessians."""
    minimum_eigenvalues = mass_weighted_hessian_eigenvalues(
        minimum_hessian_ev_per_A2, masses_amu
    )
    saddle_eigenvalues = mass_weighted_hessian_eigenvalues(
        saddle_hessian_ev_per_A2, masses_amu
    )
    return vineyard_prefactor_from_eigenvalues(
        minimum_eigenvalues,
        saddle_eigenvalues,
        zero_tol=zero_tol_rad2_per_s2,
    )


def vineyard_event_prefactors_from_forces(
    force_fn,
    min1_positions,
    saddle_positions,
    min2_positions,
    *,
    active_indices,
    masses_amu,
    step_A: float = 1.0e-3,
    zero_tol_rad2_per_s2: float = 0.0,
    progress_callback=None,
) -> VineyardEventPrefactors:
    """Compute directional Vineyard prefactors from finite-difference forces."""
    def stage_hessian(stage, positions):
        if progress_callback is not None:
            progress_callback(stage, "start")
        start = time.perf_counter()
        try:
            hessian = finite_difference_hessian_from_forces(
                force_fn,
                positions,
                active_indices,
                step_A=step_A,
            )
        except Exception:
            if progress_callback is not None:
                progress_callback(stage, "failed", time.perf_counter() - start)
            raise
        if progress_callback is not None:
            progress_callback(stage, "complete", time.perf_counter() - start)
        return hessian

    minimum_forward_hessian = stage_hessian("minimum_forward", min1_positions)
    minimum_backward_hessian = stage_hessian("minimum_backward", min2_positions)
    saddle_hessian = stage_hessian("saddle", saddle_positions)
    forward_eigenvalues = mass_weighted_hessian_eigenvalues(
        minimum_forward_hessian, masses_amu
    )
    backward_eigenvalues = mass_weighted_hessian_eigenvalues(
        minimum_backward_hessian, masses_amu
    )
    saddle_eigenvalues = mass_weighted_hessian_eigenvalues(
        saddle_hessian, masses_amu
    )
    saddle_negative = saddle_eigenvalues[saddle_eigenvalues < -zero_tol_rad2_per_s2]
    if saddle_negative.size != 1:
        raise ValueError("Vineyard saddle spectrum must have exactly one unstable mode")
    barrier_omega_rad_per_s = float(np.sqrt(abs(saddle_negative[0])))
    return VineyardEventPrefactors(
        forward_prefactor_inv_s=vineyard_prefactor_from_eigenvalues(
            forward_eigenvalues,
            saddle_eigenvalues,
            zero_tol=zero_tol_rad2_per_s2,
        ),
        backward_prefactor_inv_s=vineyard_prefactor_from_eigenvalues(
            backward_eigenvalues,
            saddle_eigenvalues,
            zero_tol=zero_tol_rad2_per_s2,
        ),
        saddle_freq_invcm=barrier_omega_rad_per_s
        / (2.0 * m.pi * SPEED_OF_LIGHT_CM_PER_S),
        barrier_omega_rad_per_s=barrier_omega_rad_per_s,
    )


def compute_rate_Eyring(dE: float, config: Config) -> float:
    r"""Compute the rate constant based on the energy barrier and parameters in the configuration.

    It uses the following equation : 
    $$
    k0*e^{-\frac{dE}{k_{b}T}}
    $$

    Parameters
    ----------
    dE : float
        The energy barrier.
    config : Config
        The configuration of the simulation.

    Returns
    -------
    float
        the rate constant.

    """
    p = PhysicalConstants()
    T = config.rateconstant.T
    k0 = config.rateconstant.k0
    return k0 * m.exp(-dE / (p.kb * T))


def _rate_model_reason(
    prefactor_source: str,
    *,
    anharmonic_corrections_active: bool,
    amsel_available: bool,
) -> str:
    if not amsel_available:
        return "amsel-unavailable"
    if prefactor_source == "rateconstant.prefactor":
        if not anharmonic_corrections_active:
            return "configured-prefactor-no-curvature"
        return "configured-prefactor"
    if not anharmonic_corrections_active:
        return "event-prefactor-no-curvature"
    return "event-prefactor-vtst"


def _rate_model_ok(reason: str) -> bool:
    return reason == "event-prefactor-vtst"


def compute_rate_amsel_vtst_details(
    dE_forward: float,
    dE_backward: float,
    config: Config,
    *,
    prefactor_inv_s: float | None = None,
    prefactor_source: str | None = None,
    saddle_freq_invcm: float | None = None,
    barrier_omega_rad_per_s: float | None = None,
) -> RateConstantDetails:
    """Return AMSEL-VTST rate value, units, correction factors, and provenance."""
    rc = config.rateconstant
    prefactor = float(
        prefactor_inv_s
        if prefactor_inv_s is not None
        else getattr(rc, "prefactor", 1.0e13)
    )
    prefactor_source = str(
        prefactor_source
        if prefactor_source is not None
        else getattr(rc, "prefactor_source", "rateconstant.prefactor")
    )
    T = float(rc.T)
    saddle_freq_invcm = float(
        saddle_freq_invcm
        if saddle_freq_invcm is not None
        else getattr(rc, "saddle_freq_invcm", 0.0)
    )
    friction_inv_s = float(getattr(rc, "friction_inv_s", 0.0))
    barrier_omega_rad_per_s = float(
        barrier_omega_rad_per_s
        if barrier_omega_rad_per_s is not None
        else getattr(rc, "barrier_omega_rad_per_s", 0.0)
    )
    correction_inputs_active = saddle_freq_invcm > 0.0 or (
        friction_inv_s > 0.0 and barrier_omega_rad_per_s > 0.0
    )
    p = PhysicalConstants()
    harmonic_rate = prefactor * m.exp(-float(dE_forward) / (p.kb * T))
    try:
        import amsel as _amsel
    except ImportError:
        reason = _rate_model_reason(
            prefactor_source,
            anharmonic_corrections_active=False,
            amsel_available=False,
        )
        return RateConstantDetails(
            style="amsel-vtst",
            rate_inv_s=harmonic_rate,
            rate_ps_inv=harmonic_rate / PS_PER_S,
            rate_unit="ps^-1",
            harmonic_rate_inv_s=harmonic_rate,
            prefactor_inv_s=prefactor,
            prefactor_source=prefactor_source,
            temperature_K=T,
            dE_forward_ev=float(dE_forward),
            dE_backward_ev=float(dE_backward),
            saddle_freq_invcm=saddle_freq_invcm,
            friction_inv_s=friction_inv_s,
            barrier_omega_rad_per_s=barrier_omega_rad_per_s,
            wigner_factor=1.0,
            eckart_factor=1.0,
            kramers_factor=1.0,
            anharmonic_corrections_active=False,
            rate_model_ok=_rate_model_ok(reason),
            rate_model_reason=reason,
        )
    result = _amsel.vtst_corrected_rate(
        prefactor,
        float(dE_forward),
        float(dE_backward),
        saddle_freq_invcm,
        T,
        friction_inv_s,
        barrier_omega_rad_per_s,
    )
    wigner = float(result.get("wigner_factor", 1.0))
    eckart = float(result.get("eckart_factor", 1.0))
    kramers = float(result.get("kramers_factor", 1.0))
    rate_inv_s = float(result["corrected_rate_inv_s"])
    anharmonic_corrections_active = correction_inputs_active and (
        wigner != 1.0 or eckart != 1.0 or kramers != 1.0 or saddle_freq_invcm > 0.0
    )
    reason = _rate_model_reason(
        prefactor_source,
        anharmonic_corrections_active=anharmonic_corrections_active,
        amsel_available=True,
    )
    return RateConstantDetails(
        style="amsel-vtst",
        rate_inv_s=rate_inv_s,
        rate_ps_inv=rate_inv_s / PS_PER_S,
        rate_unit="ps^-1",
        harmonic_rate_inv_s=float(result.get("harmonic_rate_inv_s", harmonic_rate)),
        prefactor_inv_s=prefactor,
        prefactor_source=prefactor_source,
        temperature_K=T,
        dE_forward_ev=float(dE_forward),
        dE_backward_ev=float(dE_backward),
        saddle_freq_invcm=saddle_freq_invcm,
        friction_inv_s=friction_inv_s,
        barrier_omega_rad_per_s=barrier_omega_rad_per_s,
        wigner_factor=wigner,
        eckart_factor=eckart,
        kramers_factor=kramers,
        anharmonic_corrections_active=anharmonic_corrections_active,
        rate_model_ok=_rate_model_ok(reason),
        rate_model_reason=reason,
    )


def compute_rate_amsel_vtst(
    dE_forward: float,
    dE_backward: float,
    config: Config,
    *,
    prefactor_inv_s: float | None = None,
    prefactor_source: str | None = None,
    saddle_freq_invcm: float | None = None,
    barrier_omega_rad_per_s: float | None = None,
) -> float:
    r"""Variational-TST / anharmonic-corrected rate via amsel.

    Base harmonic rate uses a PHYSICAL prefactor (``rateconstant.prefactor``,
    default ~1e13 /s) instead of the placeholder ``k0``, then multiplies by
    the amsel Wigner * Eckart * Kramers correction factors:

    $$ k = \nu\, e^{-\Delta E_f / k_b T}\, \gamma_W \gamma_E \gamma_K $$

    The Eckart factor uses the forward AND reverse barriers (asymmetry),
    both of which pyKMC carries per process. Wigner/Eckart need the saddle
    imaginary frequency (``rateconstant.saddle_freq_invcm``); when it is 0
    those factors are 1, so the style degrades to a physical-prefactor
    Eyring rate (the unphysical-prefactor fix alone). Kramers acts only when
    ``rateconstant.friction_inv_s`` > 0.

    Falls back to a physical-prefactor Eyring rate if amsel is unavailable.
    """
    return compute_rate_amsel_vtst_details(
        dE_forward,
        dE_backward,
        config,
        prefactor_inv_s=prefactor_inv_s,
        prefactor_source=prefactor_source,
        saddle_freq_invcm=saddle_freq_invcm,
        barrier_omega_rad_per_s=barrier_omega_rad_per_s,
    ).rate_ps_inv


def compute_rate(
    dE_forward: float,
    dE_backward: float,
    config: Config,
    *,
    prefactor_inv_s: float | None = None,
    prefactor_source: str | None = None,
    saddle_freq_invcm: float | None = None,
    barrier_omega_rad_per_s: float | None = None,
) -> float:
    """Dispatch the per-process rate on ``rateconstant.style``.

    - ``constant`` / ``eyring``: k0 * exp(-dE_forward / kT) (legacy).
    - ``amsel-vtst``: physical-prefactor + Wigner/Eckart/Kramers via amsel.

    ``dE_backward`` is the reverse barrier, used only by the VTST Eckart
    asymmetry factor; legacy styles ignore it.
    """
    style = getattr(config.rateconstant, "style", "constant")
    if style == "amsel-vtst":
        return compute_rate_amsel_vtst(
            dE_forward,
            dE_backward,
            config,
            prefactor_inv_s=prefactor_inv_s,
            prefactor_source=prefactor_source,
            saddle_freq_invcm=saddle_freq_invcm,
            barrier_omega_rad_per_s=barrier_omega_rad_per_s,
        )
    return compute_rate_Eyring(dE_forward, config)


def compute_htst() -> None:
    """Define a future operation to be implemented."""
    pass
