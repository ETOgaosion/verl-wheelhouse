#!/usr/bin/env bash
# Produces the two wheels that make up a `sglang-<ref>` release:
#   - sglang-kernel: the CUDA extension, compiled here from the sgl-kernel subdir
#     of the sglang submodule (scikit-build-core + CMake). This is the only
#     source build - upstream PyPI publishes sgl-kernel for a single default
#     CUDA, so building this CUDA/torch combo is the whole reason it lives here.
#   - sglang:        the main framework. Its only compiled part is a
#     CUDA-agnostic Rust frontend, and upstream already publishes portable
#     manylinux wheels, so we download-and-rehost that official wheel rather than
#     rebuild it (mirrors how flashinfer's companion wheels are vendored). This
#     keeps the release self-contained without a redundant, less-portable rebuild.
# Run with CWD = sglang/sgl-kernel.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ci/build_scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

export_extra_env
install_sgl_kernel_build_deps

# sgl-kernel's CMakeLists.txt calls cmake_policy(SET CMP0169 ...) / CMP0177 ...,
# which require CMake >= 3.31 (CMP0169 landed in 3.30, CMP0177 in 3.31) even
# though it only declares cmake_minimum_required(VERSION 3.26). We build with
# --no-build-isolation below, so scikit-build-core does NOT auto-provision a
# modern CMake from PyPI - it uses whatever `cmake` is on PATH. On ubuntu-24.04
# that's the apt CMake 3.28.3, which fails with "Policy CMP0169/CMP0177 is not
# known to this version of CMake". Pin CMake here to match sgl-kernel's own
# build recipe (sgl-kernel/Dockerfile installs CMake 3.31.1).
pip install -q "scikit-build-core>=0.10" "cmake>=3.31,<4" wheel uv ninja setuptools numpy

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

# Vendor the official prebuilt main `sglang` wheel into the same dist/ so the
# single `sglang-<ref>` release ships both wheels. The git ref (e.g. v0.5.12) is
# the sglang package version, so fetch that exact version from PyPI. A branch/SHA
# ref has no matching PyPI release, so only attempt it for version-like refs.
if [[ "${REF:-}" =~ ^v?[0-9]+\.[0-9] ]]; then
  sglang_version="${REF#v}"
  echo "::group::Vendor prebuilt sglang==${sglang_version} wheel from PyPI"
  download_prebuilt_wheel sglang "${sglang_version}" "https://pypi.org/simple" dist
  echo "::endgroup::"
else
  echo "::warning::REF='${REF:-}' is not a version tag; skipping main sglang wheel vendoring" >&2
fi

echo "Built/collected wheels:"
ls -al dist
