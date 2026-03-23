/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Core implementation of the gpumd-dlext bridge layer.
 *
 * Copyright (2026) Jaafar Mehrez <jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org>
 */

#include "gpumd_dlext.h"

#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <stdexcept>
#include <string>

#include "model/atom.cuh"
#include "model/box.cuh"

namespace gpumd_dlext {

static void dlpack_deleter(DLManagedTensor* tensor) {
    if (tensor) {
        if (tensor->dl_tensor.shape) {
            delete[] tensor->dl_tensor.shape;
        }
        if (tensor->dl_tensor.strides) {
            delete[] tensor->dl_tensor.strides;
        }
        delete tensor;
    }
}

GPUMDView::GPUMDView(Atom* atom, Box* box, int num_atoms, int device_id)
    : atom_(atom), box_(box), num_atoms_(num_atoms), device_id_(device_id), box_aligned_(nullptr) {}

GPUMDView::GPUMDView(int num_atoms)
    : atom_(nullptr), box_(nullptr), num_atoms_(num_atoms), device_id_(0), box_aligned_(nullptr) {}

GPUMDView::~GPUMDView() {
    if (box_aligned_ != nullptr) {
        cudaFreeHost(box_aligned_);
    }
}

void GPUMDView::ensure_box_buffer() {
    if (box_aligned_ == nullptr) {
        cudaMallocHost(&box_aligned_, 9 * sizeof(double));
    }
}

DLManagedTensor* GPUMDView::create_tensor(
    void* data,
    DLDataType dtype,
    std::vector<int64_t> shape,
    std::vector<int64_t> strides,
    ExecutionSpace location) {
    DLManagedTensor* tensor = new DLManagedTensor();

    tensor->dl_tensor.device.device_type =
        (location == ExecutionSpace::kOnDevice) ? kDLCUDA : kDLCPU;
    tensor->dl_tensor.device.device_id = device_id_;
    tensor->dl_tensor.data = data;
    tensor->dl_tensor.dtype = dtype;
    tensor->dl_tensor.ndim = shape.size();
    tensor->dl_tensor.shape = new int64_t[shape.size()];
    for (size_t i = 0; i < shape.size(); ++i) {
        tensor->dl_tensor.shape[i] = shape[i];
    }

    if (!strides.empty()) {
        tensor->dl_tensor.strides = new int64_t[strides.size()];
        for (size_t i = 0; i < strides.size(); ++i) {
            tensor->dl_tensor.strides[i] = strides[i];
        }
    } else {
        tensor->dl_tensor.strides = nullptr;
    }

    tensor->dl_tensor.byte_offset = 0;
    tensor->manager_ctx = nullptr;
    tensor->deleter = dlpack_deleter;

    return tensor;
}

DLManagedTensor* GPUMDView::positions(ExecutionSpace location) {
    DLDataType dtype{kDLFloat, 64, 1};
    void* data = const_cast<void*>(static_cast<const void*>(atom_->position_per_atom.data()));
    return create_tensor(data, dtype, {num_atoms_, 3}, {1, num_atoms_}, location);
}

DLManagedTensor* GPUMDView::velocities(ExecutionSpace location) {
    DLDataType dtype{kDLFloat, 64, 1};
    void* data = const_cast<void*>(static_cast<const void*>(atom_->velocity_per_atom.data()));
    return create_tensor(data, dtype, {num_atoms_, 3}, {1, num_atoms_}, location);
}

DLManagedTensor* GPUMDView::forces(ExecutionSpace location) {
    DLDataType dtype{kDLFloat, 64, 1};
    void* data = const_cast<void*>(static_cast<const void*>(atom_->force_per_atom.data()));
    return create_tensor(data, dtype, {num_atoms_, 3}, {1, num_atoms_}, location);
}

DLManagedTensor* GPUMDView::unwrapped_positions(ExecutionSpace location) {
    DLDataType dtype{kDLFloat, 64, 1};

    const double* source = nullptr;
    if (atom_ != nullptr && atom_->unwrapped_position.size() >= atom_->position_per_atom.size()) {
        source = atom_->unwrapped_position.data();
    } else {
        source = atom_->position_per_atom.data();
    }

    void* data = const_cast<void*>(static_cast<const void*>(source));
    return create_tensor(data, dtype, {num_atoms_, 3}, {1, num_atoms_}, location);
}

DLManagedTensor* GPUMDView::potential(ExecutionSpace location) {
    DLDataType dtype{kDLFloat, 64, 1};
    void* data = const_cast<void*>(static_cast<const void*>(atom_->potential_per_atom.data()));
    return create_tensor(data, dtype, {num_atoms_}, {}, location);
}

DLManagedTensor* GPUMDView::masses(ExecutionSpace location) {
    DLDataType dtype{kDLFloat, 64, 1};
    void* data = const_cast<void*>(static_cast<const void*>(atom_->mass.data()));
    return create_tensor(data, dtype, {num_atoms_}, {}, location);
}

DLManagedTensor* GPUMDView::types(ExecutionSpace location) {
    DLDataType dtype{kDLInt, 32, 1};
    void* data = const_cast<void*>(static_cast<const void*>(atom_->type.data()));
    return create_tensor(data, dtype, {num_atoms_}, {}, location);
}

DLManagedTensor* GPUMDView::box() {
    DLDataType dtype{kDLFloat, 64, 1};

    ensure_box_buffer();
    std::memcpy(box_aligned_, box_->cpu_h, 9 * sizeof(double));

    DLManagedTensor* tensor =
        create_tensor(static_cast<void*>(box_aligned_), dtype, {3, 3}, {3, 1}, ExecutionSpace::kOnHost);
    tensor->dl_tensor.device.device_type = kDLCPU;
    return tensor;
}

void GPUMDView::synchronize() {
#ifdef __CUDACC__
    cudaDeviceSynchronize();
#endif
}

std::map<std::string, int64_t> GPUMDView::tensor_info(const std::string& name) const {
    const void* data = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;

    if (name == "positions") {
        data = atom_->position_per_atom.data();
        rows = num_atoms_;
        cols = 3;
    } else if (name == "velocities") {
        data = atom_->velocity_per_atom.data();
        rows = num_atoms_;
        cols = 3;
    } else if (name == "forces") {
        data = atom_->force_per_atom.data();
        rows = num_atoms_;
        cols = 3;
    } else if (name == "unwrapped_positions") {
        data = atom_->unwrapped_position.size() >= atom_->position_per_atom.size()
            ? static_cast<const void*>(atom_->unwrapped_position.data())
            : static_cast<const void*>(atom_->position_per_atom.data());
        rows = num_atoms_;
        cols = 3;
    } else if (name == "potential") {
        data = atom_->potential_per_atom.data();
        rows = num_atoms_;
        cols = 1;
    } else if (name == "masses") {
        data = atom_->mass.data();
        rows = num_atoms_;
        cols = 1;
    } else if (name == "types") {
        data = atom_->type.data();
        rows = num_atoms_;
        cols = 1;
    } else if (name == "box") {
        data = box_aligned_ != nullptr ? static_cast<const void*>(box_aligned_) : static_cast<const void*>(box_->cpu_h);
        rows = 3;
        cols = 3;
    }

    const auto addr = reinterpret_cast<uintptr_t>(data);
    return {
        {"addr", static_cast<int64_t>(addr)},
        {"rows", rows},
        {"cols", cols},
        {"align8", static_cast<int64_t>(addr % 8)},
        {"align16", static_cast<int64_t>(addr % 16)},
        {"align32", static_cast<int64_t>(addr % 32)},
    };
}

Engine::Engine(const std::string& library_path, const std::string& run_in, const std::string& model_xyz)
    : library_handle_(nullptr),
      engine_handle_(nullptr),
      view_(nullptr),
      gpumd_init_(nullptr),
      gpumd_step_(nullptr),
      gpumd_set_forces_(nullptr),
      gpumd_compute_forces_(nullptr),
      gpumd_cleanup_(nullptr),
      gpumd_current_step_(nullptr),
      gpumd_total_steps_(nullptr),
      gpumd_num_atoms_(nullptr),
      gpumd_time_step_(nullptr),
      gpumd_atom_ptr_(nullptr),
      gpumd_box_ptr_(nullptr) {
    library_handle_ = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library_handle_ == nullptr) {
        throw std::runtime_error(std::string("Failed to load GPUMD library: ") + dlerror());
    }

    gpumd_init_ = reinterpret_cast<init_fn>(dlsym(library_handle_, "gpumd_init"));
    gpumd_step_ = reinterpret_cast<step_fn>(dlsym(library_handle_, "gpumd_step"));
    gpumd_set_forces_ = reinterpret_cast<set_forces_fn>(dlsym(library_handle_, "gpumd_set_forces"));
    gpumd_compute_forces_ = reinterpret_cast<compute_forces_fn>(dlsym(library_handle_, "gpumd_compute_forces"));
    gpumd_cleanup_ = reinterpret_cast<cleanup_fn>(dlsym(library_handle_, "gpumd_cleanup"));
    gpumd_current_step_ = reinterpret_cast<current_step_fn>(dlsym(library_handle_, "gpumd_current_step"));
    gpumd_total_steps_ = reinterpret_cast<total_steps_fn>(dlsym(library_handle_, "gpumd_total_steps"));
    gpumd_num_atoms_ = reinterpret_cast<num_atoms_fn>(dlsym(library_handle_, "gpumd_num_atoms"));
    gpumd_time_step_ = reinterpret_cast<time_step_fn>(dlsym(library_handle_, "gpumd_time_step"));
    gpumd_atom_ptr_ = reinterpret_cast<atom_ptr_fn>(dlsym(library_handle_, "gpumd_atom_ptr"));
    gpumd_box_ptr_ = reinterpret_cast<box_ptr_fn>(dlsym(library_handle_, "gpumd_box_ptr"));

    if (!gpumd_init_ || !gpumd_step_ || !gpumd_set_forces_ || !gpumd_compute_forces_ ||
        !gpumd_cleanup_ || !gpumd_current_step_ || !gpumd_total_steps_ || !gpumd_num_atoms_ ||
        !gpumd_time_step_ || !gpumd_atom_ptr_ || !gpumd_box_ptr_) {
        dlclose(library_handle_);
        throw std::runtime_error("Failed to resolve one or more required GPUMD symbols");
    }

    engine_handle_ = gpumd_init_(run_in.c_str(), model_xyz.c_str());
    if (engine_handle_ == nullptr) {
        dlclose(library_handle_);
        throw std::runtime_error("gpumd_init returned null");
    }

    view_ = std::make_unique<GPUMDView>(
        static_cast<Atom*>(gpumd_atom_ptr_(engine_handle_)),
        static_cast<Box*>(gpumd_box_ptr_(engine_handle_)),
        gpumd_num_atoms_(engine_handle_),
        0);
}

Engine::~Engine() {
    close();
}

void Engine::step() {
    gpumd_step_(engine_handle_);
}

void Engine::set_forces(const std::vector<double>& forces_xyz) {
    gpumd_set_forces_(engine_handle_, forces_xyz.data());
}

void Engine::compute_forces() {
    gpumd_compute_forces_(engine_handle_);
}

void Engine::close() {
    if (engine_handle_ != nullptr && gpumd_cleanup_ != nullptr) {
        gpumd_cleanup_(engine_handle_);
        engine_handle_ = nullptr;
    }
    view_.reset();
    if (library_handle_ != nullptr) {
        dlclose(library_handle_);
        library_handle_ = nullptr;
    }
}

int Engine::current_step() const { return gpumd_current_step_(engine_handle_); }
int Engine::total_steps() const { return gpumd_total_steps_(engine_handle_); }
int Engine::num_atoms() const { return gpumd_num_atoms_(engine_handle_); }
double Engine::time_step() const { return gpumd_time_step_(engine_handle_); }

Sampler::Sampler(GPUMDView* view) : view_(view), callback_(nullptr), callback_interval_(1) {}
Sampler::~Sampler() = default;

void Sampler::set_callback(TimestepCallback callback) {
    callback_ = std::move(callback);
}

void Sampler::set_callback_interval(int interval) {
    callback_interval_ = interval;
}

void Sampler::execute(int64_t timestep) {
    if (callback_ && callback_interval_ > 0 && timestep % callback_interval_ == 0) {
        callback_(timestep);
    }
}

static Sampler* current_sampler = nullptr;

void initialize() {}
void finalize() {}
Sampler* get_current_sampler() { return current_sampler; }

} // namespace gpumd_dlext
