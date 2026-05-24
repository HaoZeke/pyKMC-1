import importlib.util
from pathlib import Path
import tomllib


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
        "-funroll-loops -ffast-math -Ofast)\n",
        encoding="utf-8",
    )

    installer = _load_installer()
    installer._patch_native_flags(tmp_path)

    patched = cmake_file.read_text(encoding="utf-8")
    assert "-march=native" not in patched
    assert "-mtune=generic" in patched


def test_pixi_ira_feature_installs_conda_forge_package():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = data["tool"]["pixi"]["feature"]["ira"]["dependencies"]
    assert "ira" in dependencies
