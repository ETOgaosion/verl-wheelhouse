#!/usr/bin/env bash
# Builds all three flashinfer wheels: the core JIT package, the prebuilt
# cubin package, and the AOT-compiled jit-cache package (the only one of the
# three that actually compiles CUDA kernels ahead of time). Run with CWD =
# the flashinfer submodule checkout, which contains the flashinfer-cubin/
# and flashinfer-jit-cache/ subprojects.
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

echo "::group::flashinfer-cubin"
(
  cd flashinfer-cubin
  python -m build --no-isolation --wheel --outdir ../dist .
)
echo "::endgroup::"

echo "::group::flashinfer-jit-cache (AOT compile)"
(
  cd flashinfer-jit-cache
  export FLASHINFER_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
  export MAX_JOBS="${MAX_JOBS}"
  python -m build --no-isolation --wheel --outdir ../dist .
)
echo "::endgroup::"

echo "Built wheels:"
ls -al dist
