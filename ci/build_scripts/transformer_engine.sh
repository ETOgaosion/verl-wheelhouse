#!/usr/bin/env bash
# Builds the TransformerEngine PyTorch wheel using the exact env vars/flags
# verl's Dockerfiles use:
#
#   export NVTE_FRAMEWORK=pytorch && \
#     MAX_JOBS=<n> NVTE_BUILD_THREADS_PER_JOB=4 \
#     pip3 install --resume-retries 999 --no-build-isolation \
#     git+https://github.com/NVIDIA/TransformerEngine.git@${TRANSFORMER_ENGINE_VERSION}
#
# swapping `pip install` for `pip wheel --no-deps -w dist` against the local
# pinned checkout (`.`) so a distributable artifact is produced. Run with
# CWD = the TransformerEngine submodule checkout. Requires cuDNN (installed
# by the calling workflow, see versions.yaml's requires_cudnn: true).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

export_extra_env

# Both verl Dockerfiles pre-install these before building TransformerEngine.
pip install -q pybind11 nvidia-mathdx ninja wheel packaging

export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS
NVTE_CUDA_ARCHS="$(arch_list_strip_dots "${TORCH_CUDA_ARCH_LIST}")"

mkdir -p dist
MAX_JOBS="${MAX_JOBS}" pip wheel -v \
  --no-build-isolation \
  --no-deps \
  -w dist \
  --resume-retries 999 \
  .

echo "Built wheels:"
ls -al dist
