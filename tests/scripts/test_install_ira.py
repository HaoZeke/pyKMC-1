import importlib.util
from pathlib import Path


def _load_installer():
    script = Path(__file__).resolve().parents[2] / "scripts" / "install_ira.py"
    spec = importlib.util.spec_from_file_location("install_ira", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_patch_native_flags_removes_machine_specific_arch(tmp_path):
    cmake_file = tmp_path / "src" / "CMakeLists.txt"
    cmake_file.parent.mkdir()
    cmake_file.write_text(
        "add_compile_options(-ffree-line-length-none -march=native "
        "-funroll-loops -ffast-math -Ofast -xHost)\n",
        encoding="utf-8",
    )

    installer = _load_installer()
    installer._patch_native_flags(tmp_path)

    patched = cmake_file.read_text(encoding="utf-8")
    assert "-march=native" not in patched
    assert "-ffast-math" not in patched
    assert "-Ofast" not in patched
    assert "-xHost" not in patched
    assert "-mtune=generic" in patched
    assert "-O2" in patched


def test_source_dir_accepts_current_ira_pyproject_layout(tmp_path):
    (tmp_path / "interface").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        "[build-system]\nrequires = ['scikit-build-core']\n",
        encoding="utf-8",
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "project(IRA LANGUAGES Fortran C)\n",
        encoding="utf-8",
    )
    (tmp_path / "interface" / "ira_mod.py").write_text("", encoding="utf-8")

    installer = _load_installer()

    assert installer._source_dir(str(tmp_path)) == tmp_path.resolve()
