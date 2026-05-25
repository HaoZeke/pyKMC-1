from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def pypartn_interface_path_for_plugin(plugin_path) -> Path:
    plugin = Path(plugin_path).expanduser().resolve(strict=False)
    return plugin.parent.parent / "interface"


def pypartn_interface_paths(plugin_path=None) -> list[Path]:
    candidates: list[Path] = []

    if plugin_path is not None:
        candidates.append(pypartn_interface_path_for_plugin(plugin_path))

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


def add_pypartn_interface_paths(plugin_path=None) -> None:
    for candidate in reversed(pypartn_interface_paths(plugin_path=plugin_path)):
        if not (candidate / "pypARTn.py").is_file():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def import_pypartn(*, plugin_path=None):
    add_pypartn_interface_paths(plugin_path=plugin_path)
    try:
        return importlib.import_module("pypARTn")
    except ModuleNotFoundError as exc:
        if exc.name != "pypARTn":
            raise
        searched = "\n".join(
            f"  - {path}" for path in pypartn_interface_paths(plugin_path=plugin_path)
        )
        raise ModuleNotFoundError(
            "Could not import pypARTn. Provide a pARTn plugin library from an "
            "artn-plugin checkout, or set PYKMC_ARTN_PLUGIN_DIR or ARTN_PLUGIN_DIR "
            "to a checkout containing interface/pypARTn.py.\nSearched:\n"
            + searched
        ) from exc
