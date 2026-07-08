---
name: verl wheelhouse CI
overview: Build a GitHub Actions-based "wheelhouse" that compiles CUDA wheels for all 6 submodules (apex, TransformerEngine, flash-attention, flashinfer, sglang's sgl-kernel, vllm), driven by one editable/extendable Python version map, and publishes them to GitHub Releases + a GitHub Pages pip index.
todos:
  - id: versions-map
    content: Create ci/versions.py with BUILD_MATRIX, VERL_REFERENCE_VERSIONS, and COMPONENTS map
    status: completed
  - id: generate-matrix
    content: Create ci/generate_matrix.py to expand the map into GH Actions matrix JSON
    status: completed
  - id: build-scripts
    content: Create ci/build_scripts/common.sh plus per-component build scripts (apex, transformer_engine, flash_attention, flashinfer, sgl_kernel, vllm)
    status: completed
  - id: reusable-workflow
    content: Create .github/workflows/_build.yml reusable workflow mirroring flash-attention's _build.yml
    status: completed
  - id: component-workflows
    content: Create the 6 per-component trigger workflows plus build-all.yml
    status: completed
  - id: release-workflow
    content: Create release.yml (tag-triggered GH release + full matrix build/upload)
    status: completed
  - id: publish-index
    content: Create ci/build_index.py and publish-index.yml for the GitHub Pages PEP 503 index
    status: completed
  - id: docs-gitignore
    content: Write README.md documenting the map/workflows/install instructions, and update .gitignore
    status: completed
isProject: false
---

# verl-wheelhouse: multi-component CUDA wheel build/publish pipeline

## Goal
Turn this repo (currently just 6 git submodules: `apex`, `TransformerEngine`, `flashinfer`, `flash-attention`, `sglang`, `vllm`) into a wheelhouse that:
1. Builds a CUDA wheel for each submodule using the exact build commands from [`docker/Dockerfile.stable.sglang`](https://github.com/verl-project/verl/blob/main/docker/Dockerfile.stable.sglang) / [`docker/Dockerfile.stable.vllm`](https://github.com/verl-project/verl/blob/main/docker/Dockerfile.stable.vllm), on GitHub Actions structured like [flash-attention's `_build.yml`](https://github.com/Dao-AILab/flash-attention/blob/main/.github/workflows/_build.yml).
2. Uses a single, editable/extendable **Python map** (`ci/versions.py`) for the CUDA/Python/Torch build matrix and per-component version pins, seeded with CUDA `13.0.2` / Python `3.12` / Torch `2.11.0` (matching both Dockerfiles' `ARG` defaults).
3. Publishes wheels to GitHub Releases + a GitHub Pages PEP 503 "simple" index (per user's answer), on standard GitHub-hosted runners using the same resource-saving tricks flash-attention itself uses (disk cleanup, swap space, capped `MAX_JOBS`, resumable build cache) since the user confirmed "however flash-attention's own CI does it" is acceptable, including its resource constraints.

## Confirmed pinned refs for the initial matrix entry (all verified to exist on the remotes)
- CUDA `13.0.2`, Python `3.12`, Torch `2.11.0` / torchvision `0.26.0` / torchaudio `2.11.0`, `cxx11_abi=TRUE` (Torch 2.7+ default, per flash-attention's own matrix exclusions).
- `flash-attention` → tag `v2.8.3` (`FLASH_ATTENTION_VERSION` in both Dockerfiles).
- `TransformerEngine` → tag `v2.15` (`TRANSFORMER_ENGINE_VERSION`).
- `vllm` → tag `v0.23.0` (`VLLM_VERSION`).
- `apex` → branch `main` (both Dockerfiles build unpinned `git+https://github.com/NVIDIA/apex.git`).
- `flashinfer` → tag `v0.6.13` (matches `flashinfer-python==0.6.13`/`flashinfer-cubin==0.6.13` pinned in vllm's `requirements/cuda.txt`, since neither Dockerfile builds it from source).
- `sglang` (sgl-kernel only) → tag `v0.5.12` (matches `FROM lmsysorg/sglang:v0.5.12`).
- `verl_reference_versions` block also tracks `transformers=5.3.0`, `trl=0.27.0`, `nsight=2025.6.1`, `megatron=core_v0.16.1`, `verl=v0.7.1` for documentation/compatibility, even though those aren't built here.

These are all just defaults inside the editable map — bumping a version or adding a new CUDA/Python/Torch combo means editing `ci/versions.py`, no workflow changes needed.

## Why per-component build logic differs (from research of each submodule)
- **flash-attention**: plain `setup.py bdist_wheel`, env `FLASH_ATTENTION_FORCE_BUILD=TRUE`, `FLASH_ATTENTION_FORCE_CXX11_ABI`, `FLASH_ATTN_CUDA_ARCHS` (numeric, no dots, e.g. `80;90;100;120`).
- **apex**: mirror verl's `Dockerfile.stable.vllm` command exactly, just swapping `pip install` for `pip wheel -w dist` (so we get an artifact) and the source from `git+https://github.com/NVIDIA/apex.git` to the local pinned submodule checkout (`.`):
  ```bash
  MAX_JOBS=<n> pip wheel -v --no-build-isolation --no-deps -w dist \
    --disable-pip-version-check \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    .
  ```
  Has a hard `nvcc == torch.version.cuda` check (`apex/setup.py:74-89`), so the installed torch's CUDA build tag must match the toolkit we install.
- **TransformerEngine**: mirror verl's Dockerfile command exactly (same env vars/flags), again swapping to `pip wheel -w dist` against the local pinned checkout:
  ```bash
  export NVTE_FRAMEWORK=pytorch
  MAX_JOBS=<n> NVTE_BUILD_THREADS_PER_JOB=4 \
    pip wheel -v --no-build-isolation --no-deps -w dist --resume-retries 999 .
  ```
  Needs cuDNN ≥ 9.3 installed (not just the CUDA toolkit) plus `nvidia-mathdx`, `pybind11`, `ninja`, `wheel`, `packaging` pre-installed (per both Dockerfiles' `pip install pybind11 nvidia-mathdx` / `pip install wheel packaging nvidia-mathdx ninja pybind11` steps).
- **flashinfer**: JIT-first; 3 separate wheels — `python -m build` (pure JIT wheel), `flashinfer-cubin` (`python -m build --no-isolation --wheel`, downloads prebuilt cubins), `flashinfer-jit-cache` (`FLASHINFER_CUDA_ARCH_LIST=... python -m build --no-isolation --wheel`, the only one that actually AOT-compiles CUDA).
- **sglang** → only `sgl-kernel/` is compiled (scikit-build-core + CMake, `uv build --wheel --no-build-isolation` then `./rename_wheels.sh`); the main `sglang` python wheel is near-pure-python (skip building it here — it just needs Rust/protoc and doesn't have CUDA compilation, not what a "wheelhouse" needs to provide).
- **vllm**: heaviest build — `python use_existing_torch.py && pip install -r requirements/build/cuda.txt && python setup.py bdist_wheel`, env `MAX_JOBS`, `NVCC_THREADS`, `TORCH_CUDA_ARCH_LIST`, `CMAKE_BUILD_TYPE=Release`.

## New repo layout

`ci/build_scripts/common.sh` is a single shared file (not a directory) sourced by each of the six per-component build scripts alongside it. It holds only shared helper functions: disk-space cleanup, swap-space setup, cuDNN install, and one function that reformats `TORCH_CUDA_ARCH_LIST` into whichever string format each build system expects (flash-attention: `80;90;120`, apex/TE/vllm: `8.0;9.0;12.0`, flashinfer: `8.0 9.0a 12.0f`).

```
ci/
  versions.py             # THE editable/extendable map (base matrix + per-component config)
  generate_matrix.py      # reads versions.py -> emits GH Actions matrix JSON for a component (or "all")
  build_index.py          # builds a PEP 503 static index from a set of released wheel URLs
  build_scripts/
    common.sh              # shared bash helper functions, sourced by every script below
    apex.sh
    transformer_engine.sh
    flash_attention.sh
    flashinfer.sh
    sgl_kernel.sh
    vllm.sh
.github/workflows/
  _build.yml              # reusable single-combo build (mirrors flash-attention's _build.yml)
  build-apex.yml
  build-transformer-engine.yml
  build-flash-attention.yml
  build-flashinfer.yml
  build-sglang-kernel.yml
  build-vllm.yml
  build-all.yml           # convenience: builds every component x every matrix combo
  release.yml             # on `v*` tag push: creates GH release, builds+uploads all wheels as assets
  publish-index.yml       # on release published (or manual): (re)builds the GH Pages PEP503 index
README.md                 # usage docs: editing the map, triggering builds, installing from the index
.gitignore                # add dist/, build/, *.whl, __pycache__/
```

## `ci/versions.py` shape (the "map")

```python
BUILD_MATRIX = [
    dict(cuda="13.0.2", python="3.12", torch="2.11.0",
         torch_vision="0.26.0", torch_audio="2.11.0", cxx11_abi="TRUE"),
    # add more dicts here for additional CUDA/Python/Torch combos
]

VERL_REFERENCE_VERSIONS = dict(
    transformers="5.3.0", trl="0.27.0", nsight_systems="2025.6.1",
    megatron="core_v0.16.1", verl="v0.7.1",
)  # tracked for compatibility docs only; not built by this repo

COMPONENTS = {
    "flash-attention": dict(path="flash-attention", ref="v2.8.3", builder="flash_attention",
                             torch_cuda_arch_list="8.0;9.0;10.0;12.0", requires_cudnn=False,
                             max_jobs=2),
    "apex":            dict(path="apex", ref="main", builder="apex",
                             torch_cuda_arch_list="7.5;8.0;8.6;9.0;10.0;11.0;12.0", requires_cudnn=False,
                             max_jobs=2),  # verl's own Dockerfiles use MAX_JOBS=128/256 on big build boxes
    "transformer-engine": dict(path="TransformerEngine", ref="v2.15", builder="transformer_engine",
                             torch_cuda_arch_list="8.0;9.0;10.0", requires_cudnn=True,
                             nvte_build_threads_per_job=4, max_jobs=1),
    "flashinfer":      dict(path="flashinfer", ref="v0.6.13", builder="flashinfer",
                             torch_cuda_arch_list="8.0 9.0a 10.0a 12.0f", requires_cudnn=False,
                             max_jobs=2),
    "sglang-kernel":   dict(path="sglang/sgl-kernel", ref="v0.5.12", builder="sgl_kernel",
                             requires_cudnn=False, max_jobs=2),
    "vllm":            dict(path="vllm", ref="v0.23.0", builder="vllm",
                             torch_cuda_arch_list="7.5;8.0;8.6;8.9;9.0;10.0;12.0", requires_cudnn=False,
                             max_jobs=1),
}
```
`max_jobs` defaults are deliberately small (1-2) to fit standard GitHub-hosted runners, per the flash-attention-style approach — this is the same field verl's own Dockerfiles set to `128`/`256` on their large build machines, so bumping it back up is a one-line edit once builds run on bigger/self-hosted runners.

`ci/generate_matrix.py --component <name|all> --github-output` expands `COMPONENTS x BUILD_MATRIX` into the JSON list consumed by `strategy.matrix: ${{ fromJSON(...) }}`.

## `_build.yml` reusable workflow (mirrors flash-attention's `_build.yml`)
Inputs: `component, cuda-version, python-version, torch-version, torch-vision-version, torch-audio-version, cxx11-abi, upload-to-release, release-version`.
Steps (same shape as the flash-attention example, generalized across components):
1. Checkout this repo (submodules disabled), then shallow-fetch only the pinned `ref` for the given component's submodule path (so version bumps are driven purely by `ci/versions.py`, decoupled from whatever commit each submodule pointer currently has).
2. Free disk space + `pierotofy/set-swap-space` (same as flash-attention's `_build.yml`).
3. `Jimver/cuda-toolkit` to install the CUDA toolkit; if `requires_cudnn`, also apt-install `cudnn9-cuda-<major>` via the network repo (same steps both verl Dockerfiles use).
4. Install pinned `torch==<torch-version>` from the matching `download.pytorch.org/whl/cuXXX` index (+ torchvision/torchaudio for vllm).
5. Restore/save a resumable `build.tar` cache keyed by component+matrix (same trick flash-attention uses so a build that exceeds the 5h internal timeout can resume on the next run).
6. Run `ci/build_scripts/<builder>.sh` with all the matrix values exported as env vars; script `cd`s into the component path and executes that component's exact build command researched above (for apex/TE this is verl's own Dockerfile command, `pip install` swapped for `pip wheel -w dist` against the local checkout so we get an artifact), writing wheel(s) to `dist/`.
7. Upload `dist/*.whl` as a workflow artifact; if `upload-to-release`, attach each wheel to the GH release tagged `release-version` (same `actions/upload-release-asset` pattern as flash-attention).

## Publishing (per user's choice: GitHub Releases + GitHub Pages index)
- `release.yml`: on `v*` tag push, creates the GitHub release (`gh release create`, mirroring flash-attention's `setup_release` job), then runs the full `ci/generate_matrix.py --component all` matrix through `_build.yml` with `upload-to-release: true`.
- `publish-index.yml`: on `release: published` (or manual dispatch), lists all `.whl` assets across releases via `gh release list`/`gh api`, runs `ci/build_index.py` to emit a static PEP 503 `index.html` + per-package `simple/<pkg>/index.html` (linking straight to the GitHub Release asset download URLs, so Pages stays tiny), and deploys via `actions/upload-pages-artifact` + `actions/deploy-pages`. End result: `pip install --extra-index-url https://<org>.github.io/<repo>/simple/ flash-attn`.

## Caveats to document in README
- Some builds (vllm, TransformerEngine) are large even on the free runners; the resumable build-cache + swap-space + capped `MAX_JOBS`/arch-list tricks from flash-attention's own `_build.yml` are applied everywhere, but multiple re-runs may be needed to finish within GitHub's 6h job cap — this mirrors flash-attention's own tradeoffs, per the user's guidance.
- `runs-on` is a workflow input everywhere, so it's trivial to point at self-hosted/larger runners later without touching build logic.
- Only `sgl-kernel` (not the full `sglang` python package) is built from the `sglang` submodule, since that's the only piece with CUDA compilation.

## Implementation todos
