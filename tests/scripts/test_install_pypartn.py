import importlib.util
from pathlib import Path


def _load_installer():
    script = Path(__file__).resolve().parents[2] / "scripts" / "install_pypartn.py"
    spec = importlib.util.spec_from_file_location("install_pypartn", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_lammps_shim_exposes_headers_runtime_and_plugin_marker(tmp_path):
    source = tmp_path / "lammps-source"
    (source / "cmake").mkdir(parents=True)
    (source / "src").mkdir()
    (source / "src" / "version.h").write_text(
        '#define LAMMPS_VERSION "29 Aug 2024"\n',
        encoding="utf-8",
    )

    prefix = tmp_path / "prefix"
    (prefix / "lib").mkdir(parents=True)
    (prefix / "lib" / "liblammps.so").write_text("", encoding="utf-8")

    installer = _load_installer()
    shim = installer._prepare_lammps_shim(
        lammps_source=source,
        prefix=prefix,
        work_root=tmp_path / "work",
    )

    assert (shim / "cmake").exists()
    assert (shim / "src" / "version.h").exists()
    assert (shim / "src" / "liblammps.so").resolve() == prefix / "lib" / "liblammps.so"
    assert "PLUGIN" in (shim / "src" / "lmpinstalledpkgs.h").read_text(encoding="utf-8")


def test_existing_lammps_plugin_does_not_require_lammps_source(tmp_path):
    source = tmp_path / "artn-plugin"
    (source / "interface").mkdir(parents=True)
    (source / "interface" / "pypARTn.py").write_text("", encoding="utf-8")
    (source / "lib").mkdir()
    (source / "lib" / "libartn.so").write_text("", encoding="utf-8")
    (source / "lib" / "libartn-lmp.so").write_text("", encoding="utf-8")

    installer = _load_installer()

    assert not installer._needs_lammps_source(
        source,
        build=True,
        require_lammps=True,
    )
