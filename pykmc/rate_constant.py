"""Module defining function used to compute the rate constant."""

from .config import PhysicalConstants, Config
import math as m

PS_PER_S = 1.0e12


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
    rc = config.rateconstant
    prefactor = float(getattr(rc, "prefactor", 1.0e13))
    T = rc.T
    try:
        import amsel as _amsel
    except ImportError:
        p = PhysicalConstants()
        return prefactor * m.exp(-dE_forward / (p.kb * T)) / PS_PER_S
    result = _amsel.vtst_corrected_rate(
        prefactor,
        float(dE_forward),
        float(dE_backward),
        float(getattr(rc, "saddle_freq_invcm", 0.0)),
        float(T),
        float(getattr(rc, "friction_inv_s", 0.0)),
        float(getattr(rc, "barrier_omega_rad_per_s", 0.0)),
    )
    return float(result["corrected_rate_inv_s"]) / PS_PER_S


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
