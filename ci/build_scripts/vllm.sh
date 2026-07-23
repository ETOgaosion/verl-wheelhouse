#!/usr/bin/env bash
# Builds the vllm wheel following vllm's own documented "build against an
# existing torch install" flow, with the env vars verl's
# docker/Dockerfile.stable.vllm sets:
#
#   python use_existing_torch.py
#   pip install -r requirements/build/cuda.txt
#   MAX_JOBS=<n> NVCC_THREADS=<n> TORCH_CUDA_ARCH_LIST=<list> \
#     CMAKE_BUILD_TYPE=Release python setup.py bdist_wheel
#
# `use_existing_torch.py` strips the `torch==` pins out of the requirements
# files first, so the subsequent pip install doesn't clobber the pinned
# torch the calling workflow already installed. Run with CWD = the vllm
# submodule checkout.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

export_extra_env

python use_existing_torch.py
pip install -q -r requirements/build/cuda.txt

export VLLM_TARGET_DEVICE=cuda
export CMAKE_BUILD_TYPE=Release
export MAX_JOBS="${MAX_JOBS}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

ensure_cuda_nvrtc_for_cmake
# DeepGEMM's `_C` extension is compiled by tools/build_deepgemm_C.py with a
# raw g++ call that -I's the interpreter's baked (and, on self-hosted runners,
# nonexistent) sysconfig INCLUDEPY; put the real Python include dir on the
# compiler's search path so <Python.h> resolves.
ensure_python_include_path

# Pin the wheel's version to the ref we were told to build.
#
# vllm derives its version from git tags via setuptools-scm (see its
# pyproject.toml's `[tool.setuptools_scm]`). But _build.yml checks the source
# out with a shallow, tag-less fetch (`git fetch --depth 1 origin <ref>` +
# `checkout FETCH_HEAD`), so setuptools-scm can't see the pinned tag and falls
# back to its default `0.1.devN`; use_existing_torch.py above also dirties the
# tree, tacking on a `+dYYYYMMDD` local segment. The result was mislabeled
# wheels like `vllm-0.1.dev1+gee0da84ab.d20260720` that don't satisfy
# `pip install vllm==0.24.0`. vllm's setup.py exposes an explicit
# VLLM_VERSION_OVERRIDE hook (get_vllm_version() feeds it to setuptools-scm and
# returns it verbatim, before any device-suffix mangling), so set it from REF
# (exported by _build.yml). Only do this for version-like tags (e.g. v0.24.0);
# leave branch/SHA refs to setuptools-scm.
if [[ "${REF:-}" =~ ^v?[0-9]+\.[0-9] ]]; then
  export VLLM_VERSION_OVERRIDE="${REF#v}"
  echo "Pinned VLLM_VERSION_OVERRIDE=${VLLM_VERSION_OVERRIDE} (from REF=${REF})"
fi

python setup.py bdist_wheel --dist-dir=dist

echo "Built wheels:"
ls -al dist
