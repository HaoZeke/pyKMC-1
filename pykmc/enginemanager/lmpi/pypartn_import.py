from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def pypartn_interface_paths() -> list[Path]:
    candidates: list[Path] = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(
            Path(conda_prefix).expanduser()
            / "share"
            / "artn-plugin"
            / "interface"
        )

    for key in ("PYKMC_ARTN_PLUGIN_DIR", "ARTN_PLUGIN_DIR"):
        source_root = os.environ.get(key)
        if source_root:
            candidates.append(Path(source_root).expanduser() / "interface")

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if candidate in seen:
            continue
        unique_candidates.append(candidate)
        seen.add(candidate)
    return unique_candidates


def add_pypartn_interface_paths() -> None:
    for candidate in reversed(pypartn_interface_paths()):
        if not (candidate / "pypARTn.py").is_file():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def import_pypartn():
    add_pypartn_interface_paths()
    try:
        return importlib.import_module("pypARTn")
    except ModuleNotFoundError as exc:
        if exc.name != "pypARTn":
            raise
        searched = "\n".join(f"  - {path}" for path in pypartn_interface_paths())
        raise ModuleNotFoundError(
            "Could not import pypARTn. Set PYKMC_ARTN_PLUGIN_DIR or "
            "ARTN_PLUGIN_DIR to an artn-plugin checkout containing "
            "interface/pypARTn.py.\nSearched:\n"
            + searched
        ) from exc
