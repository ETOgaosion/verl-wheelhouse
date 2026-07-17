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
# install_nccl: TransformerEngine's common/util/logging.h always includes
# nccl.h, and NCCL EP (enabled when NVTE_CUDA_ARCHS contains arch >= 90)
# links libnccl at build time. Mirrors TE's own wheel Dockerfile
# (libnccl2 + libnccl-dev) and vllm/docker/Dockerfile's CUDA-matched pin.
# Expects CUDA_VERSION to be exported and the NVIDIA CUDA apt repo to already
# be configured (Jimver/cuda-toolkit's network install step does this).
# ---------------------------------------------------------------------------
install_nccl() {
  local cuda_short nccl_ver
  cuda_short="$(echo "${CUDA_VERSION}" | cut -d. -f1,2)"

  echo "::group::Install NCCL (+cuda${cuda_short})"
  sudo apt-get update
  nccl_ver="$(
    apt-cache madison libnccl-dev 2>/dev/null \
      | grep "+cuda${cuda_short}" \
      | head -1 \
      | awk -F'|' '{gsub(/^ +| +$/, "", $2); print $2}'
  )"
  if [ -z "${nccl_ver}" ]; then
    echo "::error::No libnccl-dev package found for +cuda${cuda_short}" >&2
    exit 1
  fi
  sudo apt-get install -y --no-install-recommends --allow-change-held-packages \
    "libnccl-dev=${nccl_ver}" "libnccl2=${nccl_ver}"
  echo "Installed libnccl-dev=${nccl_ver} libnccl2=${nccl_ver}"
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
  apt-get update
  apt-get install -y --no-install-recommends \
    libnuma-dev libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils pkg-config
  echo "::endgroup::"
}

# ---------------------------------------------------------------------------
# flashinfer_jit_cache_cu_index: map a CUDA toolkit version to the flashinfer.ai
# jit-cache wheel index suffix (e.g. 13.0.2 -> cu130). Cubin wheels are fetched
# from the CUDA-agnostic https://flashinfer.ai/whl index instead.
# Mirrors sglang/docker/Dockerfile and vllm/docker/Dockerfile.
# ---------------------------------------------------------------------------
flashinfer_jit_cache_cu_index() {
  echo "cu$(echo "$1" | cut -d. -f1,2 | tr -d '.')"
}

# ---------------------------------------------------------------------------
# download_flashinfer_wheel: pip download a flashinfer companion wheel with
# retry logic (flashinfer.ai can have transient network issues; jit-cache is
# ~1.2 GB). Writes the .whl into dest_dir.
# ---------------------------------------------------------------------------
download_flashinfer_wheel() {
  local package="$1"
  local version="$2"
  local index_url="$3"
  local dest_dir="$4"
  local attempt max_attempts=5

  for attempt in $(seq 1 "${max_attempts}"); do
    if pip download "${package}==${version}" \
      --index-url "${index_url}" \
      --no-deps \
      -d "${dest_dir}"; then
      echo "Downloaded ${package}==${version} from ${index_url}"
      return 0
    fi
    if [ "${attempt}" -lt "${max_attempts}" ]; then
      echo "::warning::Attempt ${attempt}/${max_attempts} to download ${package} failed; retrying in 10s..."
      sleep 10
    fi
  done

  echo "::error::Failed to download ${package}==${version} from ${index_url} after ${max_attempts} attempts" >&2
  return 1
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
# ensure_cuda_cccl_include_path: CUDA 13+ moved CCCL headers (cuda/atomic,
# cuda/std/*, thrust, cub) under ${CUDA_HOME}/include/cccl/. nvcc adds this
# automatically, but host C++ TUs built with g++ (e.g. mscclpp inside
# sgl-kernel) need CPLUS_INCLUDE_PATH / C_INCLUDE_PATH set explicitly.
# Mirrors sgl-kernel Dockerfile and flash-attention hopper/setup.py.
# ---------------------------------------------------------------------------
ensure_cuda_cccl_include_path() {
  local cuda_home cccl_include="" candidate env_var current

  cuda_home="${CUDA_HOME:-}"
  if [ -z "${cuda_home}" ] && command -v nvcc >/dev/null 2>&1; then
    cuda_home="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi
  cuda_home="${cuda_home:-/usr/local/cuda}"

  for candidate in \
    "${cuda_home}/include/cccl" \
    "${cuda_home}/targets/x86_64-linux/include/cccl"; do
    if [ -d "${candidate}/cuda" ]; then
      cccl_include="${candidate}"
      break
    fi
  done

  if [ -z "${cccl_include}" ]; then
    echo "::warning::CCCL include dir not found under ${cuda_home}; CUDA 13+ host builds may fail" >&2
    return 0
  fi

  for env_var in CPLUS_INCLUDE_PATH C_INCLUDE_PATH; do
    current="${!env_var:-}"
    if [[ ":${current}:" == *":${cccl_include}:"* ]]; then
      continue
    fi
    if [ -n "${current}" ]; then
      export "${env_var}=${cccl_include}:${current}"
    else
      export "${env_var}=${cccl_include}"
    fi
  done
  echo "Prepended ${cccl_include} to CPLUS_INCLUDE_PATH and C_INCLUDE_PATH"
}

# ---------------------------------------------------------------------------
# ensure_python_include_path: vllm's tools/build_deepgemm_C.py compiles
# DeepGEMM's pybind11 `_C` extension with a bare `g++` invocation whose only
# Python header dir comes from the target interpreter's
# sysconfig.get_config_var('INCLUDEPY') (see cmake/external_projects/
# deepgemm.cmake). GitHub's actions/setup-python CPython builds bake an
# absolute INCLUDEPY of /opt/hostedtoolcache/Python/<ver>/x64/include/... into
# _sysconfigdata at build time; on self-hosted runners (where the tool cache
# actually lives under /actions-runner/_work/_tool/...) that baked path does
# not exist, so g++ dies with "fatal error: Python.h: No such file or
# directory" even though the headers ship right next to the interpreter.
# sys.prefix/base_prefix - and therefore sysconfig.get_path('include') - are
# recomputed from the interpreter's on-disk location at startup, so they still
# resolve correctly; export that real, existing include dir on
# CPLUS_INCLUDE_PATH/C_INCLUDE_PATH so g++ finds Python.h regardless of the
# stale -I INCLUDEPY flag build_deepgemm_C.py passes.
#
# Prefer-first-existing order: get_path("include") (derived from the runtime-
# resolved sys.base_prefix, and normally correct) > base_prefix/prefix guesses
# > INCLUDEPY (the stale build-time value, kept last purely as a fallback). The
# probe runs via `python -c` as a bash single-quoted string, so its Python body
# uses double quotes only - an apostrophe there would prematurely close the
# shell string (and a heredoc nested in $() mis-parses the same way).
# ---------------------------------------------------------------------------
ensure_python_include_path() {
  local py_inc env_var current
  py_inc="$(python -c '
import os, sys, sysconfig
ver = sysconfig.get_python_version()
abi = sysconfig.get_config_var("abiflags") or ""
candidates = [
    sysconfig.get_path("include"),
    os.path.join(sys.base_prefix, "include", "python" + ver + abi),
    os.path.join(sys.prefix, "include", "python" + ver + abi),
    sysconfig.get_config_var("INCLUDEPY"),
]
seen = set()
for c in candidates:
    if not c or c in seen:
        continue
    seen.add(c)
    if os.path.exists(os.path.join(c, "Python.h")):
        print(c)
        break
')"

  if [ -z "${py_inc}" ]; then
    echo "::warning::Could not locate a Python include dir containing Python.h; DeepGEMM _C build may fail" >&2
    return 0
  fi

  for env_var in CPLUS_INCLUDE_PATH C_INCLUDE_PATH; do
    current="${!env_var:-}"
    if [[ ":${current}:" == *":${py_inc}:"* ]]; then
      continue
    fi
    if [ -n "${current}" ]; then
      export "${env_var}=${py_inc}:${current}"
    else
      export "${env_var}=${py_inc}"
    fi
  done
  echo "Prepended ${py_inc} to CPLUS_INCLUDE_PATH and C_INCLUDE_PATH"
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
