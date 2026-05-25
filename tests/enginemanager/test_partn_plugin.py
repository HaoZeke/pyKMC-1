import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pykmc.enginemanager.lmpi import lammps_operations


class FakeLammps:
    def __init__(self, registering_paths=None):
        self.commands = []
        self.registering_paths = set(registering_paths or [])
        self.artn_registered = False

    def has_style(self, category, name):
        return category == "fix" and name == "artn" and self.artn_registered

    def command(self, command):
        self.commands.append(command)
        for path in self.registering_paths:
            if command == f"plugin load {path}":
                self.artn_registered = True


class FakeEngine:
    def __init__(self, lammps):
        self.lmp = lammps
        self.engine_id = 0
        self.rank = 1

    def command(self, command):
        self.lmp.command(command)


def test_lammps_operations_import_discovers_pypartn_from_plugin_checkout(
    tmp_path, monkeypatch
):
    plugin_root = tmp_path / "artn-plugin"
    interface = plugin_root / "interface"
    lib = plugin_root / "lib"
    interface.mkdir(parents=True)
    lib.mkdir()
    (interface / "pypARTn.py").write_text("MARKER = 'env-plugin'\n", encoding="utf-8")
    (lib / "libartn-lmp.so").write_text("", encoding="utf-8")

    env = os.environ.copy()
    env["PYKMC_ARTN_PLUGIN_DIR"] = str(plugin_root)
    env.pop("ARTN_PLUGIN_DIR", None)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pykmc.enginemanager.lmpi.lammps_operations; "
            "import pypARTn; print(pypARTn.MARKER)",
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "env-plugin"


def test_load_partn_plugin_uses_configured_plugin_when_registered(tmp_path):
    plugin = tmp_path / "configured" / "libartn-lmp.so"
    plugin.parent.mkdir()
    plugin.write_text("", encoding="utf-8")

    lammps = FakeLammps(registering_paths={str(plugin)})
    engine = FakeEngine(lammps)
    config = SimpleNamespace(partn=SimpleNamespace(path_artnso=str(plugin)))

    loaded = lammps_operations.load_partn_plugin(engine, config)

    assert loaded == str(plugin)
    assert lammps.has_style("fix", "artn")
    assert lammps.commands == [f"plugin load {plugin}"]


def test_load_partn_plugin_uses_conda_installed_plugin_when_configured_path_fails(
    tmp_path, monkeypatch
):
    configured = tmp_path / "missing" / "libartn-lmp.so"
    installed = tmp_path / "prefix" / "share" / "artn-plugin" / "lib" / "libartn-lmp.so"
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "prefix"))

    lammps = FakeLammps(registering_paths={str(installed)})
    engine = FakeEngine(lammps)
    config = SimpleNamespace(partn=SimpleNamespace(path_artnso=str(configured)))

    loaded = lammps_operations.load_partn_plugin(engine, config)

    assert loaded == str(installed)
    assert lammps.has_style("fix", "artn")
    assert lammps.commands == [f"plugin load {installed}"]


def test_load_partn_plugin_raises_when_loaded_plugin_does_not_register_artn(tmp_path):
    plugin = tmp_path / "configured" / "libartn-lmp.so"
    plugin.parent.mkdir()
    plugin.write_text("", encoding="utf-8")

    engine = FakeEngine(FakeLammps())
    config = SimpleNamespace(partn=SimpleNamespace(path_artnso=str(plugin)))

    with pytest.raises(RuntimeError, match="fix artn"):
        lammps_operations.load_partn_plugin(engine, config)


def test_partn_search_configures_artn_evaluation_limit(monkeypatch):
    created_artn = []
    import_plugin_paths = []

    class FakeArtn:
        def __init__(self, engine):
            self.engine = engine
            self.settings = []

        def reset_input(self):
            self.settings.append(("reset_input", None))

        def set(self, name, value):
            self.settings.append((name, value))

    class FakePypartn:
        @staticmethod
        def artn(engine):
            artn = FakeArtn(engine)
            created_artn.append(artn)
            return artn

    def fake_import_pypartn(*, plugin_path=None):
        import_plugin_paths.append(plugin_path)
        return FakePypartn

    engine = FakeEngine(FakeLammps())
    config = SimpleNamespace(
        control=SimpleNamespace(active_volume=False),
        eventsearch=SimpleNamespace(delr_thr=0.5),
        atomicenvironment=SimpleNamespace(rcut=6.5),
        partn=SimpleNamespace(
            path_artnso="/unused/libartn-lmp.so",
            dmax=6.0,
            verbosity=2,
            delr_thr=0.1,
            zseed=1000,
            push_mode="rad",
            push_dist_thr=1.0,
            push_step_size=0.4,
            ninit=2,
            lanczos_min_size=10,
            lanczos_max_size=20,
            lanczos_disp=0.0005,
            lanczos_eval_conv_thr=0.001,
            eigval_thr=-0.01,
            eigen_step_size=0.2,
            nsmooth=3,
            neigen=1,
            alpha_mix_cr=0.2,
            nnewchance=0,
            nperp=3,
            nperp_limitation=None,
            forc_thr=0.001,
            convergence_property="norm",
            nevalf_max=37,
            push_over=1.0,
            evalf_max=9999,
        ),
    )
    monkeypatch.setattr(
        lammps_operations, "load_partn_plugin", lambda *_args: config.partn.path_artnso
    )
    monkeypatch.setattr(lammps_operations, "import_pypartn", fake_import_pypartn)

    lammps_operations.partn_search(engine, config, central_atom_idx=4)

    assert import_plugin_paths == [config.partn.path_artnso]
    assert ("converge_property", "norm") in created_artn[0].settings
    assert ("nevalf_max", 37) in created_artn[0].settings
    assert "minimize 1e-6 1e-8 10000 38" in engine.lmp.commands
