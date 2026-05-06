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

    def command(self, command):
        self.lmp.command(command)


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
