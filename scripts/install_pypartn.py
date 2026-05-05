from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys


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


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), f"(cwd={cwd})")
    subprocess.run(cmd, cwd=cwd, check=True)


def _build_artn(source: Path) -> None:
    build_dir = source / "build-pykmc-pypartn"
    _run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=source,
    )
    _run(["cmake", "--build", str(build_dir)], cwd=source)


def _site_packages() -> Path:
    paths = site.getsitepackages()
    if not paths:
        raise SystemExit("No site-packages directory reported by this Python.")
    return Path(paths[0]).resolve()


def install(source: Path, build: bool) -> None:
    if build and not (source / "lib" / "libartn.so").exists():
        _build_artn(source)

    wrapper = source / "interface" / "pypARTn.py"
    base_lib = source / "lib" / "libartn.so"
    lammps_lib = source / "lib" / "libartn-lmp.so"
    if not wrapper.exists():
        raise SystemExit(f"Missing pypARTn wrapper: {wrapper}")
    if not base_lib.exists():
        raise SystemExit(
            f"Missing {base_lib}. Build artn-plugin first or pass --build."
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
    args = parser.parse_args()
    install(_source_dir(args.source), build=args.build)


if __name__ == "__main__":
    main()
