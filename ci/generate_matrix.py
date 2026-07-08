#!/usr/bin/env python3
"""Expand versions.yaml (in the project's base directory) into a GitHub
Actions build matrix.

versions.yaml is the single editable/extendable source of truth for CUDA /
Python / Torch combinations and per-component build config - it is plain
data (no Python), so it can be read and edited without touching any code.
This script is the only place that turns it into something GitHub Actions
can consume.

Every entry in the emitted JSON list is a flat dict that maps 1:1 onto the
inputs of .github/workflows/_build.yml, so a workflow can do:

    strategy:
      matrix:
        include: ${{ fromJSON(needs.compute-matrix.outputs.matrix) }}

Usage:
    python ci/generate_matrix.py --component vllm
    python ci/generate_matrix.py --component all
    python ci/generate_matrix.py --list-components
    python ci/generate_matrix.py --component vllm --github-output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# versions.yaml lives in the project's base directory (one level up from ci/).
VERSIONS_FILE = Path(__file__).resolve().parent.parent / "versions.yaml"


def load_versions() -> Dict[str, Any]:
    with open(VERSIONS_FILE, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def component_names(versions: Dict[str, Any]) -> List[str]:
    return sorted(versions["components"])


def get_component(versions: Dict[str, Any], name: str) -> Dict[str, Any]:
    try:
        return versions["components"][name]
    except KeyError as exc:
        known = ", ".join(component_names(versions))
        raise SystemExit(f"Unknown component {name!r}. Known components: {known}") from exc


def build_matrix_entries(versions: Dict[str, Any], component: str) -> List[Dict[str, Any]]:
    """Cartesian-product one component's config against the whole build_matrix."""
    cfg = get_component(versions, component)
    entries = []
    for combo in versions["build_matrix"]:
        entries.append(
            {
                "component": component,
                "path": cfg["path"],
                "ref": str(cfg["ref"]),
                "builder": cfg["builder"],
                "runs_on": cfg["runs_on"],
                "cuda": str(combo["cuda"]),
                "python": str(combo["python"]),
                "torch": str(combo["torch"]),
                "torch_vision": str(combo["torch_vision"]),
                "torch_audio": str(combo["torch_audio"]),
                "cxx11_abi": str(combo["cxx11_abi"]),
                "torch_cuda_arch_list": cfg.get("torch_cuda_arch_list") or "",
                "requires_cudnn": "true" if cfg.get("requires_cudnn") else "false",
                "max_jobs": str(cfg.get("max_jobs", 1)),
                # Already a JSON *string*; the calling workflow passes it
                # straight through to _build.yml's extra-env input (do not
                # re-encode it with toJSON() in the workflow, or it will be
                # double-escaped).
                "extra_env": json.dumps(cfg.get("extra_env") or {}, sort_keys=True),
            }
        )
    return entries


def build_full_matrix(versions: Dict[str, Any], components: List[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for name in components:
        entries.extend(build_matrix_entries(versions, name))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--component",
        default="all",
        help="Component name from versions.yaml, or 'all' (default) for every component.",
    )
    parser.add_argument(
        "--list-components",
        action="store_true",
        help="Print known component names (one per line) and exit.",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Also write matrix=<json> to $GITHUB_OUTPUT, for use in a workflow step.",
    )
    args = parser.parse_args()

    versions = load_versions()

    if args.list_components:
        print("\n".join(component_names(versions)))
        return

    if args.component == "all":
        components = component_names(versions)
    else:
        get_component(versions, args.component)  # validates, raises a clear SystemExit if unknown
        components = [args.component]

    matrix = build_full_matrix(versions, components)
    payload = json.dumps(matrix)
    print(payload)

    if args.github_output:
        github_output = os.environ.get("GITHUB_OUTPUT")
        if not github_output:
            raise SystemExit("--github-output requires $GITHUB_OUTPUT to be set")
        with open(github_output, "a", encoding="utf-8") as fh:
            fh.write(f"matrix={payload}\n")


if __name__ == "__main__":
    main()
