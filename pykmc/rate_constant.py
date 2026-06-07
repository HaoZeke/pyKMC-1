"""Module defining function used to compute the rate constant."""

from dataclasses import dataclass

from .config import PhysicalConstants, Config
import math as m
import numpy as np

PS_PER_S = 1.0e12
EV_J = 1.602176634e-19
ANGSTROM_M = 1.0e-10
AMU_KG = 1.66053906660e-27
EV_PER_A2_AMU_TO_RAD2_PER_S2 = EV_J / (ANGSTROM_M**2 * AMU_KG)


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
    dE_forward: float, dE_backward: float, config: Config
) -> RateConstantDetails:
    """Return AMSEL-VTST rate value, units, correction factors, and provenance."""
    rc = config.rateconstant
    prefactor = float(getattr(rc, "prefactor", 1.0e13))
    prefactor_source = str(getattr(rc, "prefactor_source", "rateconstant.prefactor"))
    T = float(rc.T)
    saddle_freq_invcm = float(getattr(rc, "saddle_freq_invcm", 0.0))
    friction_inv_s = float(getattr(rc, "friction_inv_s", 0.0))
    barrier_omega_rad_per_s = float(getattr(rc, "barrier_omega_rad_per_s", 0.0))
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
    dE_forward: float, dE_backward: float, config: Config
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
        dE_forward, dE_backward, config
    ).rate_ps_inv


def compute_rate(dE_forward: float, dE_backward: float, config: Config) -> float:
    """Dispatch the per-process rate on ``rateconstant.style``.

    - ``constant`` / ``eyring``: k0 * exp(-dE_forward / kT) (legacy).
    - ``amsel-vtst``: physical-prefactor + Wigner/Eckart/Kramers via amsel.

    ``dE_backward`` is the reverse barrier, used only by the VTST Eckart
    asymmetry factor; legacy styles ignore it.
    """
    style = getattr(config.rateconstant, "style", "constant")
    if style == "amsel-vtst":
        return compute_rate_amsel_vtst(dE_forward, dE_backward, config)
    return compute_rate_Eyring(dE_forward, config)


def compute_htst() -> None:
    """Define a future operation to be implemented."""
    pass
