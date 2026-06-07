from __future__ import annotations

import argparse
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


def _smoke_test() -> bool:
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
    result = subprocess.run([sys.executable, "-c", code], cwd=Path.cwd())
    return result.returncode == 0


def install(source: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="pykmc-ira-") as tmp:
        work = _copy_source(source, Path(tmp))
        _patch_native_flags(work)
        _install(work)
    if not _smoke_test():
        raise SystemExit("IRA installation failed its matcher smoke test.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Path to an IRA checkout.")
    args = parser.parse_args()
    if args.source is None and _smoke_test():
        return
    try:
        source = _source_dir(args.source)
    except SystemExit as exc:
        if args.source is None:
            raise SystemExit(
                "Installed IRA package failed its matcher smoke test and no "
                "source checkout was available for a local rebuild.\n" + str(exc)
            ) from exc
        raise
    install(source)


if __name__ == "__main__":
    main()
