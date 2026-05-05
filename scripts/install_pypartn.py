from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
import tempfile


def _source_dir(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for key in ("PYKMC_ARTN_PLUGIN_DIR", "ARTN_PLUGIN_DIR"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value))
    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repo_root / "external" / "artn-plugin",
            repo_root.parent / "artn-plugin",
        ]
    )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "interface" / "pypARTn.py").exists():
            return candidate
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise SystemExit(
        "Could not find artn-plugin source. Set PYKMC_ARTN_PLUGIN_DIR to a "
        "checkout containing interface/pypARTn.py.\nSearched:\n" + searched
    )


def _conda_prefix() -> Path:
    value = os.environ.get("CONDA_PREFIX")
    if not value:
        raise SystemExit("CONDA_PREFIX is not set; run inside a Pixi environment.")
    return Path(value).resolve()


def _lammps_source_dir(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for key in ("PYKMC_LAMMPS_SOURCE_DIR", "LAMMPS_SOURCE_DIR"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value))

    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repo_root / "external" / "lammps",
            repo_root.parent / "lammps",
        ]
    )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "src" / "version.h").exists() and (
            candidate / "cmake"
        ).is_dir():
            return candidate

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise SystemExit(
        "Could not find LAMMPS source headers. Set PYKMC_LAMMPS_SOURCE_DIR "
        "to a checkout containing src/version.h and cmake/.\nSearched:\n"
        + searched
    )


def _prepare_lammps_shim(lammps_source: Path, prefix: Path, work_root: Path) -> Path:
    liblammps = prefix / "lib" / "liblammps.so"
    if not liblammps.exists():
        raise SystemExit(f"Missing LAMMPS runtime library: {liblammps}")

    shim = work_root / "lammps-shim"
    src = shim / "src"
    shim.mkdir(parents=True, exist_ok=False)
    os.symlink(lammps_source / "cmake", shim / "cmake", target_is_directory=True)
    shutil.copytree(lammps_source / "src", src, symlinks=True)
    os.symlink(liblammps, src / "liblammps.so")

    installed_packages = src / "lmpinstalledpkgs.h"
    if not installed_packages.exists():
        installed_packages.write_text("#define LMP_PLUGIN\n", encoding="utf-8")
    elif "PLUGIN" not in installed_packages.read_text(encoding="utf-8"):
        with installed_packages.open("a", encoding="utf-8") as handle:
            handle.write("\n#define LMP_PLUGIN\n")

    return shim


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), f"(cwd={cwd})")
    subprocess.run(cmd, cwd=cwd, check=True)


def _build_artn(source: Path, *, lammps_source: Path | None = None) -> None:
    build_dir = source / "build-pykmc-pypartn"
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    target = ["cmake", "--build", str(build_dir)]
    if lammps_source is not None:
        with tempfile.TemporaryDirectory(prefix="pykmc-lammps-shim-") as tmp:
            shim = _prepare_lammps_shim(
                lammps_source=lammps_source,
                prefix=_conda_prefix(),
                work_root=Path(tmp),
            )
            configure.extend(["-DWITH_LAMMPS=ON", f"-DLAMMPS_PATH={shim}"])
            _run(configure, cwd=source)
            _run(target + ["--target", "artn-lmp"], cwd=source)
        return
    _run(configure, cwd=source)
    _run(target, cwd=source)


def _site_packages() -> Path:
    paths = site.getsitepackages()
    if not paths:
        raise SystemExit("No site-packages directory reported by this Python.")
    return Path(paths[0]).resolve()


def install(
    source: Path,
    build: bool,
    require_lammps: bool,
    lammps_source: Path | None,
) -> None:
    base_lib = source / "lib" / "libartn.so"
    lammps_lib = source / "lib" / "libartn-lmp.so"
    if build and (not base_lib.exists() or (require_lammps and not lammps_lib.exists())):
        _build_artn(source, lammps_source=lammps_source if require_lammps else None)

    wrapper = source / "interface" / "pypARTn.py"
    if not wrapper.exists():
        raise SystemExit(f"Missing pypARTn wrapper: {wrapper}")
    if not base_lib.exists():
        raise SystemExit(
            f"Missing {base_lib}. Build artn-plugin first or pass --build."
        )
    if require_lammps and not lammps_lib.exists():
        raise SystemExit(
            f"Missing {lammps_lib}. Build the LAMMPS plugin with "
            "--require-lammps and PYKMC_LAMMPS_SOURCE_DIR."
        )

    prefix = _conda_prefix()
    install_root = prefix / "share" / "artn-plugin"
    interface_dir = install_root / "interface"
    lib_dir = install_root / "lib"
    interface_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(wrapper, interface_dir / "pypARTn.py")
    shutil.copy2(base_lib, lib_dir / "libartn.so")
    if lammps_lib.exists():
        shutil.copy2(lammps_lib, lib_dir / "libartn-lmp.so")

    pth = _site_packages() / "pykmc-pypartn.pth"
    pth.write_text(str(interface_dir) + "\n", encoding="utf-8")
    print(f"Installed pypARTn interface into {interface_dir}")
    print(f"Installed pARTn libraries into {lib_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Path to an artn-plugin checkout.")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build libartn.so with CMake when the source tree has no built library.",
    )
    parser.add_argument(
        "--require-lammps",
        action="store_true",
        help="Require and build libartn-lmp.so for LAMMPS-backed pARTn calls.",
    )
    parser.add_argument("--lammps-source", help="Path to a LAMMPS source checkout.")
    args = parser.parse_args()
    lammps_source = (
        _lammps_source_dir(args.lammps_source) if args.require_lammps else None
    )
    install(
        _source_dir(args.source),
        build=args.build,
        require_lammps=args.require_lammps,
        lammps_source=lammps_source,
    )


if __name__ == "__main__":
    main()
