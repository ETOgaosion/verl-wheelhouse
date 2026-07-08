#!/usr/bin/env bash
# Builds the flash-attention wheel, mirroring the build step in
# flash-attention's own .github/workflows/_build.yml. Run with CWD = the
# flash-attention submodule checkout.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

export_extra_env

pip install -q ninja packaging wheel psutil setuptools

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_FORCE_CXX11_ABI="${CXX11_ABI}"
export FLASH_ATTN_CUDA_ARCHS
FLASH_ATTN_CUDA_ARCHS="$(arch_list_strip_dots "${TORCH_CUDA_ARCH_LIST}")"
export MAX_JOBS="${MAX_JOBS}"
export NVCC_THREADS="${NVCC_THREADS:-2}"

python setup.py bdist_wheel --dist-dir=dist

echo "Built wheels:"
ls -al dist
