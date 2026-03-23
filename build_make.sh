#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Build script for gpumd-dlext using nvcc and a compatible GPUMD source tree.
#
# Copyright (2026) Jaafar Mehrez <jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUMD_ROOT="${GPUMD_ROOT:-${1:-}}"
BUILD_DIR="${SCRIPT_DIR}/build"
PACKAGE_DIR="${SCRIPT_DIR}/python/gpumd_dlext"
PYTHON_BIN="${PYTHON:-python3}"
CXX="${CXX:-g++}"
NVCC="${NVCC:-nvcc}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_INC="${CUDA_HOME}/include"
CUDA_LIB="${CUDA_HOME}/lib64"

if [ -z "${GPUMD_ROOT}" ]; then
    echo "Error: GPUMD_ROOT is not set."
    echo "Usage: GPUMD_ROOT=/path/to/GPUMD ./build_make.sh"
    echo "   or: ./build_make.sh /path/to/GPUMD"
    exit 1
fi

if [ ! -d "${GPUMD_ROOT}/src" ]; then
    echo "Error: GPUMD source tree not found at ${GPUMD_ROOT}"
    exit 1
fi

mkdir -p "${BUILD_DIR}" "${PACKAGE_DIR}"

NVCCFLAGS="-O3 -I${SCRIPT_DIR}/include -I${GPUMD_ROOT}/src -I${CUDA_INC} -Xcompiler -fPIC --expt-relaxed-constexpr"

${NVCC} ${NVCCFLAGS} -x cu -c "${SCRIPT_DIR}/src/gpumd_dlext.cpp" -o "${BUILD_DIR}/gpumd_dlext.o"
${NVCC} -shared -o "${BUILD_DIR}/libgpumd_dlext.so" "${BUILD_DIR}/gpumd_dlext.o" -L"${CUDA_LIB}" -lcudart -ldl

PYBIND11_INCLUDES="$(${PYTHON_BIN} -m pybind11 --includes)"
${CXX} -O3 -Wall -shared -std=c++14 -fPIC ${PYBIND11_INCLUDES} \
    "${SCRIPT_DIR}/python/gpumd_dlext_binding.cpp" \
    "${BUILD_DIR}/libgpumd_dlext.so" \
    -o "${BUILD_DIR}/_gpumd_dlext.so" \
    -I"${SCRIPT_DIR}/include" \
    -I"${GPUMD_ROOT}/src" \
    -I"${CUDA_INC}" \
    -L"${CUDA_LIB}" -lcudart -ldl

cp "${BUILD_DIR}/_gpumd_dlext.so" "${PACKAGE_DIR}/"
cp "${BUILD_DIR}/libgpumd_dlext.so" "${PACKAGE_DIR}/"

cd "${SCRIPT_DIR}"
"${PYTHON_BIN}" -m pip install -e python/

echo "Build and install complete."
echo "Runtime use still requires a compatible GPUMD libgpumd.so build."
