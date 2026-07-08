# verl-wheelhouse

A GitHub Actions-based "wheelhouse" that builds prebuilt CUDA wheels for the
heavy native-extension dependencies of [verl](https://github.com/verl-project/verl):

| Component | Submodule path | What gets built |
|---|---|---|
| [apex](https://github.com/NVIDIA/apex) | `apex` | full apex (`--cpp_ext --cuda_ext`) |
| [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) | `TransformerEngine` | PyTorch TE extension |
| [flash-attention](https://github.com/Dao-AILab/flash-attention) | `flash-attention` | `flash_attn` |
| [flashinfer](https://github.com/flashinfer-ai/flashinfer) | `flashinfer` | `flashinfer-python`, `flashinfer-cubin`, `flashinfer-jit-cache` |
| [sglang](https://github.com/sgl-project/sglang) | `sglang/sgl-kernel` | `sgl-kernel` only (the rest of `sglang` is pure Python) |
| [vllm](https://github.com/vllm-project/vllm) | `vllm` | `vllm` |

Build commands follow the exact flags used in verl's own
[`docker/Dockerfile.stable.sglang`](https://github.com/verl-project/verl/blob/main/docker/Dockerfile.stable.sglang)
and
[`docker/Dockerfile.stable.vllm`](https://github.com/verl-project/verl/blob/main/docker/Dockerfile.stable.vllm),
and the GitHub Actions structure mirrors flash-attention's own
[`_build.yml`](https://github.com/Dao-AILab/flash-attention/blob/main/.github/workflows/_build.yml)
reusable workflow, including its resource-saving tricks for standard
GitHub-hosted runners (disk cleanup, swap space, capped parallelism,
resumable build caches).

Wheels are published to [GitHub Releases](../../releases) and to a static
[GitHub Pages](https://pages.github.com/) PEP 503 "simple" package index, so
they're directly `pip install`-able.

## Repo layout

```
versions.yaml             # THE editable/extendable version map (see below), kept in the project's base dir
ci/
  generate_matrix.py      # expands versions.yaml into a GH Actions matrix
  build_index.py          # builds the static PEP 503 index from release assets
  build_scripts/
    common.sh              # shared bash helpers, sourced by every script below
    apex.sh
    transformer_engine.sh
    flash_attention.sh
    flashinfer.sh
    sgl_kernel.sh
    vllm.sh
.github/workflows/
  _build.yml              # reusable single-combination build workflow
  build-apex.yml
  build-transformer-engine.yml
  build-flash-attention.yml
  build-flashinfer.yml
  build-sglang-kernel.yml
  build-vllm.yml
  build-all.yml           # builds every component x every matrix combo
  release.yml             # on `v*` tag push: GH release + build + upload
  publish-index.yml       # (re)publishes the GitHub Pages PEP 503 index
```

## The version map (`versions.yaml`)

Everything version-related lives in one editable, plain-data YAML file at
the project's base directory - no Python, no workflow YAML - needs to
change for routine version bumps:

- **`build_matrix`**: the CUDA / Python / Torch combinations to build. Seeded
  with CUDA `13.0.2`, Python `3.12`, Torch `2.11.0` (matching both verl
  Dockerfiles' `ARG` defaults). Add another entry to build more combinations
  - every component is built once per entry here.
- **`components`**: per-submodule config - the git `ref` to build (branch,
  tag, or commit; overrides whatever commit the submodule pointer in this
  repo is on), which `ci/build_scripts/<builder>.sh` to run, the CUDA arch
  list, whether cuDNN is required, `max_jobs`, the runner label, and any
  extra environment variables.
- **`verl_reference_versions`**: versions verl's Dockerfiles pin for things
  this repo does *not* build (`transformers`, `trl`, `nsight_systems`,
  `megatron`, `verl` itself) - tracked here purely for compatibility
  bookkeeping.

To add a new CUDA/Python/Torch combination, append an entry to
`build_matrix`. To bump a component's version, edit its `ref`. To add a
brand-new component, add an entry to `components` and a matching
`ci/build_scripts/<builder>.sh`.

`ci/generate_matrix.py` is the only code that reads `versions.yaml`; it
turns it into the flat JSON matrix GitHub Actions consumes. Inspect the
resulting matrix locally at any time (requires `pip install pyyaml`):

```bash
python3 ci/generate_matrix.py --list-components
python3 ci/generate_matrix.py --component vllm | python3 -m json.tool
python3 ci/generate_matrix.py --component all
```

## Triggering builds

- Each component has its own workflow (`build-<component>.yml`) that can be
  run on demand from the Actions tab (`workflow_dispatch`), and also runs
  automatically on pushes to `main` that touch `versions.yaml`,
  `ci/generate_matrix.py`, `ci/build_scripts/common.sh`, or that component's
  build script.
- `build-all.yml` builds every component and also runs on a weekly schedule
  as a sanity sweep.
- Pushing a tag matching `v*` runs `release.yml`, which creates (or reuses)
  the matching GitHub release, builds the full matrix, uploads every wheel
  as a release asset, and finishes by republishing the package index.
- `publish-index.yml` can also be run standalone (or fires automatically
  whenever a release is published/edited/deleted) to refresh the index
  without rebuilding any wheels.

## Installing built wheels

Once GitHub Pages is enabled for this repo (Settings → Pages → Source:
GitHub Actions) and at least one release has been published:

```bash
pip install --extra-index-url https://<owner>.github.io/<repo>/simple/ flash-attn
pip install --extra-index-url https://<owner>.github.io/<repo>/simple/ vllm
```

Or install a specific wheel directly from a [release](../../releases) page.

## Caveats

- **Build time on free runners.** `vllm` and `TransformerEngine` in
  particular are large builds even with capped `MAX_JOBS`/arch lists. The
  reusable workflow caps each build attempt at 5 hours (under GitHub's 6h
  job limit) and saves a resumable build-cache tarball on timeout, so
  re-running the workflow continues from where it left off - the same
  tradeoff flash-attention's own CI makes.
- **`runs-on` is a per-component input** (`versions.yaml`'s `runs_on`
  field), so pointing a component at a bigger/self-hosted runner (and
  bumping its `max_jobs` back up towards verl's own `128`/`256`) is a
  one-line edit, no workflow changes required.
- **Only `sgl-kernel` is built from the `sglang` submodule.** The rest of
  the `sglang` Python package has no CUDA to compile, so it's out of scope
  for this wheelhouse.
- **`apex` tracks `main`** (both verl Dockerfiles build it unpinned); every
  other component tracks a specific tag.
