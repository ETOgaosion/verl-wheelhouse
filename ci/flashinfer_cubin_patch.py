"""
CI hook for flashinfer-cubin wheel builds (loaded via flashinfer_cubin_patch.pth).

When FLASHINFER_CUBINS_PRELOADED=1, skip download_artifacts() inside the PEP 517
build backend so wheel packaging does not re-fetch ~16k cubins.
"""

from __future__ import annotations

import builtins
import functools
import os
import sys
from pathlib import Path
from types import ModuleType


def _flashinfer_root() -> Path | None:
    cwd = Path.cwd()
    for root in (cwd.parent, cwd):
        if (root / "flashinfer" / "artifacts.py").is_file():
            return root
    return None


def _apply_artifacts_patch(artifacts: ModuleType) -> None:
    if os.environ.get("FLASHINFER_CUBINS_PRELOADED") != "1":
        return
    if getattr(artifacts.download_artifacts, "_verl_wheelhouse_patched", False):
        return

    def skip_preloaded_download() -> None:
        print(
            "flashinfer-cubin: cubins preloaded by CI, skipping download_artifacts()"
        )

    skip_preloaded_download._verl_wheelhouse_patched = True  # type: ignore[attr-defined]
    artifacts.download_artifacts = skip_preloaded_download


def _install_import_hook() -> None:
    if os.environ.get("FLASHINFER_CUBINS_PRELOADED") != "1":
        return

    root = _flashinfer_root()
    if root is None:
        return

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    _orig_import = builtins.__import__

    @functools.wraps(_orig_import)
    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = _orig_import(name, globals, locals, fromlist, level)
        if name == "flashinfer.artifacts" or (
            fromlist and "artifacts" in fromlist and name == "flashinfer"
        ):
            target = sys.modules.get("flashinfer.artifacts")
            if target is not None:
                _apply_artifacts_patch(target)
        return module

    builtins.__import__ = _patched_import


_install_import_hook()
