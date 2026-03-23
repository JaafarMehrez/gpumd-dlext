<!--
SPDX-License-Identifier: GPL-3.0-or-later
Copyright (2026) Jaafar Mehrez <jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org>
-->

# gpumd-dlext

`gpumd-dlext` exposes GPUMD simulation state as DLPack tensors so Python tools such as PySAGES and JAX can access GPUMD data with minimal copying.

## Status

This repository contains the bridge layer only. It does not vendor GPUMD or PySAGES.

At the moment, `gpumd-dlext` requires a compatible GPUMD source tree and runtime library build that provide:

- `libgpumd.so`
- the GPUMD driver C API used by the bridge
- driver-controlled stepping and external-force injection support

Those GPUMD capabilities currently live in development work and are expected to be proposed upstream separately. Until then, this repository should be treated as depending on a patched GPUMD branch or fork.

## Scope

This repository contains:

- the C++ DLPack bridge
- Python bindings
- build scripts

## Build Requirements

- CUDA toolkit
- Python 3.8+
- pybind11
- a compatible GPUMD source tree

## Building

Point the build at a compatible GPUMD source tree with `GPUMD_ROOT`:

```bash
GPUMD_ROOT=/path/to/GPUMD ./build_make.sh
```

Or:

```bash
./build_make.sh /path/to/GPUMD
```

## Runtime Requirement

Building `gpumd-dlext` against GPUMD headers is not sufficient by itself. Runtime use also requires a compatible `libgpumd.so` produced from the same GPUMD branch or commit family.

## Recommended Compatibility Policy

Until GPUMD and PySAGES support are upstreamed, document and pin the exact dependency revisions you test against, for example:

- GPUMD fork/branch/commit
- PySAGES fork/branch/commit
- CUDA version
- JAX version

## Planned Upstream Work

The long-term plan is:

1. upstream the required GPUMD library and driver API support
2. upstream the PySAGES GPUMD DLPack backend support
3. keep `gpumd-dlext` focused on the bridge layer and packaging

## Notes

The bridge currently includes GPUMD internal headers from the GPUMD source tree. That means ABI and source compatibility should be treated as version-coupled until the GPUMD-facing API is narrowed further.
