/**
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Public declarations for the gpumd-dlext bridge layer.
 *
 * Copyright (2026) Jaafar Mehrez <jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org>
 */

#ifndef GPUMD_DLEXT_H
#define GPUMD_DLEXT_H

#include "dlpack.h"
#include <cuda_runtime.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

class Atom;
class Box;
class Integrate;
class Force;

namespace gpumd_dlext {

enum class ExecutionSpace {
    kOnHost,
    kOnDevice
};

class GPUMDView {
public:
    GPUMDView(Atom* atom, Box* box, int num_atoms, int device_id = 0);
    GPUMDView(int num_atoms);
    ~GPUMDView();

    DLManagedTensor* positions(ExecutionSpace location = ExecutionSpace::kOnDevice);
    DLManagedTensor* velocities(ExecutionSpace location = ExecutionSpace::kOnDevice);
    DLManagedTensor* forces(ExecutionSpace location = ExecutionSpace::kOnDevice);
    DLManagedTensor* unwrapped_positions(ExecutionSpace location = ExecutionSpace::kOnDevice);
    DLManagedTensor* potential(ExecutionSpace location = ExecutionSpace::kOnDevice);
    DLManagedTensor* masses(ExecutionSpace location = ExecutionSpace::kOnDevice);
    DLManagedTensor* types(ExecutionSpace location = ExecutionSpace::kOnDevice);
    DLManagedTensor* box();

    void synchronize();
    int num_atoms() const { return num_atoms_; }
    int device_id() const { return device_id_; }
    std::map<std::string, int64_t> tensor_info(const std::string& name) const;

private:
    Atom* atom_;
    Box* box_;
    int num_atoms_;
    int device_id_;
    double* box_aligned_;

    DLManagedTensor* create_tensor(
        void* data,
        DLDataType dtype,
        std::vector<int64_t> shape,
        std::vector<int64_t> strides,
        ExecutionSpace location
    );
    void ensure_box_buffer();
};

class Engine {
public:
    Engine(const std::string& library_path, const std::string& run_in, const std::string& model_xyz);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void step();
    void set_forces(const std::vector<double>& forces_xyz);
    void compute_forces();
    void close();

    int current_step() const;
    int total_steps() const;
    int num_atoms() const;
    double time_step() const;

    GPUMDView* view() { return view_.get(); }

private:
    void* library_handle_;
    void* engine_handle_;
    std::unique_ptr<GPUMDView> view_;

    using init_fn = void* (*)(const char*, const char*);
    using step_fn = void (*)(void*);
    using set_forces_fn = void (*)(void*, const double*);
    using compute_forces_fn = void (*)(void*);
    using cleanup_fn = void (*)(void*);
    using current_step_fn = int (*)(const void*);
    using total_steps_fn = int (*)(const void*);
    using num_atoms_fn = int (*)(const void*);
    using time_step_fn = double (*)(const void*);
    using atom_ptr_fn = void* (*)(void*);
    using box_ptr_fn = void* (*)(void*);

    init_fn gpumd_init_;
    step_fn gpumd_step_;
    set_forces_fn gpumd_set_forces_;
    compute_forces_fn gpumd_compute_forces_;
    cleanup_fn gpumd_cleanup_;
    current_step_fn gpumd_current_step_;
    total_steps_fn gpumd_total_steps_;
    num_atoms_fn gpumd_num_atoms_;
    time_step_fn gpumd_time_step_;
    atom_ptr_fn gpumd_atom_ptr_;
    box_ptr_fn gpumd_box_ptr_;
};

using TimestepCallback = std::function<void(int64_t timestep)>;

class Sampler {
public:
    explicit Sampler(GPUMDView* view);
    ~Sampler();

    void set_callback(TimestepCallback callback);
    void set_callback_interval(int interval);
    void execute(int64_t timestep);
    GPUMDView* view() { return view_; }

private:
    GPUMDView* view_;
    TimestepCallback callback_;
    int callback_interval_;
};

void initialize();
void finalize();
Sampler* get_current_sampler();

} // namespace gpumd_dlext

#endif
