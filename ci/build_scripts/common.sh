#!/usr/bin/env bash
# Shared helpers sourced by every ci/build_scripts/<component>.sh script (and,
# for free_disk_space/install_cudnn, by .github/workflows/_build.yml itself).
#
# The calling workflow is expected to have already exported the relevant
# matrix fields as environment variables before invoking a build script, e.g.
# CUDA_VERSION, TORCH_CUDA_ARCH_LIST, MAX_JOBS, EXTRA_ENV (see _build.yml's
# "Build wheel" step). This file must be *sourced*, not executed.

set -euo pipefail

# ---------------------------------------------------------------------------
# free_disk_space: same cleanup flash-attention's own _build.yml runs before
# a CUDA build, to claw back space on standard GitHub-hosted runners.
# ---------------------------------------------------------------------------
free_disk_space() {
  echo "::group::Free up disk space"
  sudo rm -rf /usr/share/dotnet || true
  sudo rm -rf /opt/ghc || true
  sudo rm -rf /opt/hostedtoolcache/CodeQL || true
  sudo rm -rf /usr/local/lib/android || true
  echo "::endgroup::"
}

# ---------------------------------------------------------------------------
# install_cudnn: mirrors the cuDNN network-repo install both verl Dockerfiles
# run before building TransformerEngine. Expects CUDA_VERSION to be exported.
# ---------------------------------------------------------------------------
install_cudnn() {
  local cuda_major arch
  cuda_major="$(echo "${CUDA_VERSION}" | cut -d. -f1)"
  arch="$(uname -m)"
  if [ "${arch}" = "aarch64" ]; then
    arch="sbsa"
  fi

  echo "::group::Install cuDNN ${cuda_major}"
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${arch}/cuda-keyring_1.1-1_all.deb"
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  rm -f cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install -y --allow-downgrades --allow-change-held-packages \
    "cudnn9-cuda-${cuda_major}"
  echo "::endgroup::"
}

# ---------------------------------------------------------------------------
# arch_list_strip_dots: convert the canonical "8.0;9.0;10.0;12.0" arch list
# into the undotted "80;90;100;120" form that flash-attention's
# FLASH_ATTN_CUDA_ARCHS and TransformerEngine's NVTE_CUDA_ARCHS expect.
# apex/vllm consume TORCH_CUDA_ARCH_LIST in the canonical dotted form
# directly (no conversion needed), and flashinfer's list is given
# pre-formatted in versions.yaml (with PTX-family suffix letters like
# "9.0a"/"12.0f") since those can't be derived mechanically.
# ---------------------------------------------------------------------------
arch_list_strip_dots() {
  echo "$1" | tr -d '.'
}

# ---------------------------------------------------------------------------
# install_sgl_kernel_build_deps: sgl-kernel pulls in mscclpp via CMake, which
# requires libnuma and libibverbs dev headers/libs. Mirrors sglang's CI apt
# install list (scripts/ci/cuda/ci_install_dependency.sh) and the sgl-kernel
# Dockerfile (numactl-devel, libibverbs).
# ---------------------------------------------------------------------------
install_sgl_kernel_build_deps() {
  echo "::group::Install sgl-kernel build dependencies"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    libnuma-dev libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils pkg-config
  echo "::endgroup::"
}

# ---------------------------------------------------------------------------
# export_extra_env: EXTRA_ENV is exported as a JSON object string by the
# workflow (e.g. '{"NVTE_BUILD_THREADS_PER_JOB": "4"}'); turn its entries
# into real exported environment variables.
# ---------------------------------------------------------------------------
export_extra_env() {
  if [ -z "${EXTRA_ENV:-}" ] || [ "${EXTRA_ENV}" = "{}" ]; then
    return 0
  fi
  local key value
  while IFS='=' read -r key value; do
    [ -n "${key}" ] || continue
    export "${key}=${value}"
    echo "Exported ${key}=${value} (from extra_env)"
  done < <(echo "${EXTRA_ENV}" | jq -r 'to_entries[] | "\(.key)=\(.value)"')
}

# ---------------------------------------------------------------------------
# ensure_cuda_nvrtc_for_cmake: CMake FindCUDAToolkit looks for an unversioned
# libnvrtc.so, but the cuda-nvrtc apt sub-package on GitHub runners only ships
# libnvrtc.so.<major> (see vllm-project/vllm#29669). Point CMake at the
# versioned library explicitly via CMAKE_ARGS (used by vllm's setup.py and
# sgl-kernel's scikit-build-core/uv build).
# ---------------------------------------------------------------------------
ensure_cuda_nvrtc_for_cmake() {
  if [[ "${CMAKE_ARGS:-}" == *"CUDA_nvrtc_LIBRARY"* ]]; then
    return 0
  fi

  local cuda_home libdir nvrtc
  cuda_home="${CUDA_HOME:-}"
  if [ -z "${cuda_home}" ] && command -v nvcc >/dev/null 2>&1; then
    cuda_home="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi
  cuda_home="${cuda_home:-/usr/local/cuda}"

  for libdir in "${cuda_home}/lib64" "${cuda_home}/targets/x86_64-linux/lib"; do
    [ -d "${libdir}" ] || continue
    if [ -e "${libdir}/libnvrtc.so" ]; then
      nvrtc="${libdir}/libnvrtc.so"
      break
    fi
    nvrtc="$(find "${libdir}" -maxdepth 1 -name 'libnvrtc.so.*' -type f -print 2>/dev/null | sort -V | tail -1)"
    [ -n "${nvrtc}" ] && break
  done

  if [ -z "${nvrtc}" ]; then
    echo "::warning::Could not locate libnvrtc under ${cuda_home}; vllm CMake may fail" >&2
    return 0
  fi

  export CMAKE_ARGS="${CMAKE_ARGS:-} -DCUDA_nvrtc_LIBRARY=${nvrtc}"
  echo "Set CMAKE_ARGS CUDA_nvrtc_LIBRARY=${nvrtc}"
}
