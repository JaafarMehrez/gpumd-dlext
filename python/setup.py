# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (2026) Jaafar Mehrez <jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org>

from setuptools import find_packages, setup

setup(
    name="gpumd_dlext",
    version="0.1.0",
    description="Wraps GPUMD simulation data as DLPack tensors for PySAGES/JAX integration",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
