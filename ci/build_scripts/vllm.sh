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

python setup.py bdist_wheel --dist-dir=dist

echo "Built wheels:"
ls -al dist
