import sys
from types import SimpleNamespace

from pykmc.rate_constant import compute_rate_amsel_vtst


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
