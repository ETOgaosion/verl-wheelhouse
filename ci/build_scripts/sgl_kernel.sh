#!/usr/bin/env bash
# Builds the sgl-kernel CUDA extension wheel out of the sglang submodule
# (scikit-build-core + CMake). Only sgl-kernel is built from sglang - the
# main `sglang` python package has no CUDA to compile, so it isn't part of
# this wheelhouse. Run with CWD = sglang/sgl-kernel.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

export_extra_env
install_sgl_kernel_build_deps

pip install -q "scikit-build-core>=0.10" wheel uv ninja setuptools numpy

export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}"

ensure_cuda_nvrtc_for_cmake
ensure_cuda_cccl_include_path

# sgl-kernel defaults SGL_KERNEL_COMPILE_THREADS=32 (nvcc --threads per TU).
# On GitHub's 7 GB runners, MAX_JOBS parallel ninja targets × 32 nvcc threads
# routinely OOM-kills the build (exit 143). sgl-kernel's own README recommends
# CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1" for memory-constrained builds.
export CMAKE_ARGS="${CMAKE_ARGS:-} -DSGL_KERNEL_COMPILE_THREADS=1 -DGITHUB_ARTIFACTORY=github.com"
echo "CMAKE_ARGS=${CMAKE_ARGS}"

uv build --wheel --no-build-isolation -Cbuild-dir=build .
./rename_wheels.sh

echo "Built wheels:"
ls -al dist
