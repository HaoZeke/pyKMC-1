from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def _source_dir(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for key in ("PYKMC_IRA_DIR", "IRA_DIR"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value))

    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repo_root / "external" / "ira",
            repo_root.parent / "IterativeRotationsAssignments",
            repo_root.parent / "ira",
        ]
    )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (
            (candidate / "setup.py").exists()
            and (candidate / "interface" / "ira_mod.py").exists()
            and (candidate / "src" / "CMakeLists.txt").exists()
        ):
            return candidate

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise SystemExit(
        "Could not find IRA source. Set PYKMC_IRA_DIR to a checkout containing "
        "setup.py, interface/ira_mod.py, and src/CMakeLists.txt.\nSearched:\n"
        + searched
    )


def _source_install_requested(explicit: str | None) -> bool:
    return bool(
        explicit
        or os.environ.get("PYKMC_IRA_DIR")
        or os.environ.get("IRA_DIR")
    )


def _installed_package_available() -> bool:
    return importlib.util.find_spec("ira_mod") is not None


def _copy_source(source: Path, work_root: Path) -> Path:
    work = work_root / "ira"
    shutil.copytree(
        source,
        work,
        ignore=shutil.ignore_patterns(
            ".git",
            "build",
            "build_cmake",
            "dist",
            "*.egg-info",
            "__pycache__",
        ),
    )
    stale_lib = work / "lib" / "libira.so"
    if stale_lib.exists():
        stale_lib.unlink()
    return work


def _patch_native_flags(source: Path) -> None:
    cmake_file = source / "src" / "CMakeLists.txt"
    text = cmake_file.read_text(encoding="utf-8")
    patched = text.replace("-march=native", "-mtune=generic")
    if patched == text:
        return
    cmake_file.write_text(patched, encoding="utf-8")


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), f"(cwd={cwd})")
    subprocess.run(cmd, cwd=cwd, check=True)


def _install(work: Path) -> None:
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-build-isolation",
            "--no-deps",
            ".",
        ],
        cwd=work,
    )


def _smoke_test() -> None:
    code = (
        "import ira_mod, numpy as np; "
        "coords=np.array([[0., 0., 0.], [1., 0., 0.], "
        "[0., 1., 0.], [0., 0., 1.]], dtype=float); "
        "coords2=coords + np.array([0.1, 0.2, 0.3]); "
        "rmat, tr, perm, dh = ira_mod.IRA().match("
        "4, ['X'] * 4, coords, 4, ['X'] * 4, coords2, 2.0"
        "); "
        "print('IRA OK', float(dh))"
    )
    _run([sys.executable, "-c", code], cwd=Path.cwd())


def install(source: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="pykmc-ira-") as tmp:
        work = _copy_source(source, Path(tmp))
        _patch_native_flags(work)
        _install(work)
    _smoke_test()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Path to an IRA checkout.")
    args = parser.parse_args()
    if (
        not _source_install_requested(args.source)
        and _installed_package_available()
    ):
        _smoke_test()
        return
    install(_source_dir(args.source))


if __name__ == "__main__":
    main()
