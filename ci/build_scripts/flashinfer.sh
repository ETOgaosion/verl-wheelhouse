#!/usr/bin/env bash
# Builds flashinfer wheels for the wheelhouse:
#   - flashinfer-python: compiled from the pinned flashinfer submodule checkout
#   - flashinfer-cubin:    prebuilt wheel from https://flashinfer.ai/whl
#   - flashinfer-jit-cache: prebuilt wheel from https://flashinfer.ai/whl/cu<XY>
#
# Mirrors sglang/docker/Dockerfile "PARALLEL STAGE 3: FlashInfer Cache" and verl's
# Dockerfiles: cubin is CUDA-version-agnostic; jit-cache is fetched from the
# CUDA-specific flashinfer.ai index.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

export_extra_env

pip install -q build wheel ninja numpy

mkdir -p dist

echo "::group::flashinfer-python (JIT core wheel)"
python -m build --wheel --outdir dist .
echo "::endgroup::"

FLASHINFER_VERSION="$(tr -d '[:space:]' < version.txt)"
FLASHINFER_CU_INDEX="$(flashinfer_jit_cache_cu_index "${CUDA_VERSION}")"
FLASHINFER_CUBIN_INDEX="https://flashinfer.ai/whl"
FLASHINFER_JIT_CACHE_INDEX="https://flashinfer.ai/whl/${FLASHINFER_CU_INDEX}"

echo "flashinfer version=${FLASHINFER_VERSION} cuda=${CUDA_VERSION} jit-cache index=${FLASHINFER_JIT_CACHE_INDEX}"

echo "::group::flashinfer-cubin (prebuilt wheel)"
download_flashinfer_wheel \
  "flashinfer-cubin" \
  "${FLASHINFER_VERSION}" \
  "${FLASHINFER_CUBIN_INDEX}" \
  dist
echo "::endgroup::"

echo "::group::flashinfer-jit-cache (prebuilt wheel)"
download_flashinfer_wheel \
  "flashinfer-jit-cache" \
  "${FLASHINFER_VERSION}" \
  "${FLASHINFER_JIT_CACHE_INDEX}" \
  dist
echo "::endgroup::"

echo "Built wheels:"
ls -al dist
