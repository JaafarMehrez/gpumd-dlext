# SPDX-License-Identifier: GPL-3.0-or-later
# gpumd-dlext build configuration. 
#
# Copyright (2026) Jaafar Mehrez <jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org>


cmake_minimum_required(VERSION 3.18)
project(gpumd-dlext VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(GPUMD_ROOT "" CACHE PATH "Path to a GPUMD source tree with libgpumd support")
if(NOT GPUMD_ROOT)
    message(FATAL_ERROR "GPUMD_ROOT must be set to a compatible GPUMD source tree.")
endif()

if(NOT EXISTS "${GPUMD_ROOT}/src")
    message(FATAL_ERROR "GPUMD_ROOT does not look like a GPUMD source tree: ${GPUMD_ROOT}")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GPUMD_ROOT}/src
    ${GPUMD_ROOT}/src/utilities
    ${GPUMD_ROOT}/src/model
    ${CUDAToolkit_INCLUDE_DIRS}
)

add_library(gpumd_dlext SHARED
    src/gpumd_dlext.cpp
)

target_link_libraries(gpumd_dlext
    CUDA::cudart
    ${CMAKE_DL_LIBS}
)

target_include_directories(gpumd_dlext PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)

    pybind11_add_module(_gpumd_dlext
        python/gpumd_dlext_binding.cpp
    )

    target_link_libraries(_gpumd_dlext PRIVATE
        gpumd_dlext
        CUDA::cudart
        ${CMAKE_DL_LIBS}
    )

    target_include_directories(_gpumd_dlext PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GPUMD_ROOT}/src
    )

    set_target_properties(_gpumd_dlext PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/gpumd_dlext
    )
endif()

install(TARGETS gpumd_dlext
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

enable_testing()
add_test(NAME test_import
    COMMAND ${Python3_EXECUTABLE} -c "import gpumd_dlext"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
)
