---
name: manage-wheelhouse-components
description: >-
  Upgrade a pinned component version or add a brand-new CUDA wheel component
  to the verl-wheelhouse build pipeline. Covers editing versions.yaml,
  writing ci/build_scripts/<builder>.sh, and creating/copying a
  .github/workflows/build-<component>.yml trigger workflow. Use when asked
  to bump/update/pin a submodule version, add a new wheel/component to the
  wheelhouse, or edit versions.yaml in this repo.
---

# Managing verl-wheelhouse components

Everything is driven by `versions.yaml` (repo root, plain YAML data). For
the full step-by-step guide with a worked example, see
[docs/maintaining-components.md](../../../docs/maintaining-components.md).
This file has the quick-reference version.

## Upgrading an existing component's version

1. Edit the component's `ref:` under `components:` in `versions.yaml`.
2. If upstream changed supported GPU architectures, update
   `torch_cuda_arch_list` too (see arch-list conventions below).
3. Validate: `pip install pyyaml && python3 ci/generate_matrix.py --component <name>`.
4. Commit. Pushing to `main` auto-triggers `build-<component>.yml` (its
   `paths:` filter matches `versions.yaml`).

`apex` tracks `main` unpinned (matches verl's own Dockerfiles) - there is no
version to bump for it.

## Adding a brand-new component

Checklist:

- [ ] `git submodule add <url> <path>`
- [ ] Add a `components:` entry to `versions.yaml` (`path`, `ref`,
      `builder`, `torch_cuda_arch_list`, `requires_cudnn`, `max_jobs`,
      `runs_on`, `extra_env`) - follow the schema comments already in that
      file.
- [ ] Create `ci/build_scripts/<builder>.sh`. Copy the shape of an existing
      script (`ci/build_scripts/vllm.sh` is a good default): shebang,
      `set -euo pipefail`, source `common.sh`, call `export_extra_env`,
      install prerequisite pip packages, run the project's own documented
      wheel-build command (mirror its own CI/Dockerfile exactly), leave the
      wheel(s) in `dist/` relative to CWD. Then `chmod +x` it.
- [ ] Copy an existing `.github/workflows/build-<component>.yml` (e.g.
      `build-vllm.yml`) to `build-<new-component>.yml`; update its `name:`,
      `paths:` filter entries, and the `--component <name>` argument.
- [ ] Update `README.md`'s component table and repo-layout listing.
- [ ] Validate:
      `pip install pyyaml && python3 ci/generate_matrix.py --component <name>`
      and `bash -n ci/build_scripts/<builder>.sh`.
- [ ] No changes needed in `build-all.yml` or `release.yml` - both already
      use `--component all` and pick up new components automatically.

## Arch-list conventions

`torch_cuda_arch_list` in `versions.yaml` is canonical dotted+semicolon form
(e.g. `8.0;9.0;12.0`); each build script converts it as needed:

- apex, vllm: used as-is.
- flash-attention, TransformerEngine: undotted via `common.sh`'s
  `arch_list_strip_dots` (e.g. `80;90;120`).
- flashinfer: given verbatim with PTX-family suffixes (e.g.
  `8.0 9.0a 12.0f`) since those can't be derived mechanically.
- sgl-kernel: set to `null` - it hardcodes its own gencode flags.

## Key invariant

Every build script must leave the final `.whl` file(s) in `dist/` relative
to its own CWD (the component's checkout path) - `_build.yml` uploads
`<path>/dist/*.whl` as both a workflow artifact and (on release) a GitHub
Release asset.
