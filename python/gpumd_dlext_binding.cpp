/**
 * SPDX-License-Identifier: GPL-3.0-or-later
 * pybind11 bindings for gpumd-dlext.
 *
 * Copyright (2026) Jaafar Mehrez <jaafarmehrez@sjtu.edu.cn/jaafar@hpqc.org>
 */

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/dlpack.h"
#include "../include/gpumd_dlext.h"

namespace py = pybind11;
using namespace gpumd_dlext;

namespace {

py::capsule to_dlpack_capsule(DLManagedTensor* tensor) {
    return py::capsule(static_cast<void*>(tensor), "dltensor", [](PyObject* obj) {
        py::capsule capsule(obj, "dltensor");
        if (DLManagedTensor* tensor = static_cast<DLManagedTensor*>(capsule.get_pointer())) {
            if (tensor->deleter) {
                tensor->deleter(tensor);
            }
        }
    });
}

class DLPackTensor {
public:
    explicit DLPackTensor(DLManagedTensor* tensor) : tensor_(tensor) {}
    DLPackTensor(const DLPackTensor&) = delete;
    DLPackTensor& operator=(const DLPackTensor&) = delete;
    DLPackTensor(DLPackTensor&& other) noexcept : tensor_(other.tensor_) {
        other.tensor_ = nullptr;
    }

    py::capsule dlpack(py::object stream = py::none()) {
        (void)stream;
        DLManagedTensor* tensor = tensor_;
        tensor_ = nullptr;
        return to_dlpack_capsule(tensor);
    }

    py::tuple dlpack_device() const {
        if (tensor_ == nullptr) {
            throw std::runtime_error("DLPack tensor has already been consumed");
        }
        return py::make_tuple(
            static_cast<int>(tensor_->dl_tensor.device.device_type),
            tensor_->dl_tensor.device.device_id);
    }

    ~DLPackTensor() {
        if (tensor_ != nullptr && tensor_->deleter) {
            tensor_->deleter(tensor_);
        }
    }

private:
    DLManagedTensor* tensor_;
};

py::object wrap_tensor(DLManagedTensor* tensor) {
    if (!tensor) {
        return py::none();
    }
    return py::cast(new DLPackTensor(tensor), py::return_value_policy::take_ownership);
}

} // namespace

PYBIND11_MODULE(_gpumd_dlext, m) {
    m.doc() = "gpumd-dlext Python bindings";

    py::enum_<ExecutionSpace>(m, "ExecutionSpace")
        .value("kOnHost", ExecutionSpace::kOnHost)
        .value("kOnDevice", ExecutionSpace::kOnDevice);

    py::class_<DLPackTensor>(m, "DLPackTensor")
        .def("__dlpack__", &DLPackTensor::dlpack, py::arg("stream") = py::none())
        .def("__dlpack_device__", &DLPackTensor::dlpack_device);

    py::class_<GPUMDView>(m, "GPUMDView")
        .def(py::init<int>(), py::arg("num_atoms"))
        .def("positions", [](GPUMDView& self, ExecutionSpace location) {
            return wrap_tensor(self.positions(location));
        }, py::arg("location") = ExecutionSpace::kOnDevice)
        .def("velocities", [](GPUMDView& self, ExecutionSpace location) {
            return wrap_tensor(self.velocities(location));
        }, py::arg("location") = ExecutionSpace::kOnDevice)
        .def("forces", [](GPUMDView& self, ExecutionSpace location) {
            return wrap_tensor(self.forces(location));
        }, py::arg("location") = ExecutionSpace::kOnDevice)
        .def("unwrapped_positions", [](GPUMDView& self, ExecutionSpace location) {
            return wrap_tensor(self.unwrapped_positions(location));
        }, py::arg("location") = ExecutionSpace::kOnDevice)
        .def("potential", [](GPUMDView& self, ExecutionSpace location) {
            return wrap_tensor(self.potential(location));
        }, py::arg("location") = ExecutionSpace::kOnDevice)
        .def("masses", [](GPUMDView& self, ExecutionSpace location) {
            return wrap_tensor(self.masses(location));
        }, py::arg("location") = ExecutionSpace::kOnDevice)
        .def("types", [](GPUMDView& self, ExecutionSpace location) {
            return wrap_tensor(self.types(location));
        }, py::arg("location") = ExecutionSpace::kOnDevice)
        .def("box", [](GPUMDView& self) { return wrap_tensor(self.box()); })
        .def("tensor_info", &GPUMDView::tensor_info)
        .def("synchronize", &GPUMDView::synchronize)
        .def("num_atoms", &GPUMDView::num_atoms)
        .def("device_id", &GPUMDView::device_id);

    py::class_<Engine>(m, "Engine")
        .def(py::init<const std::string&, const std::string&, const std::string&>(),
             py::arg("library_path"),
             py::arg("run_in"),
             py::arg("model_xyz"))
        .def("step", &Engine::step)
        .def("set_forces", &Engine::set_forces)
        .def("compute_forces", &Engine::compute_forces)
        .def("close", &Engine::close)
        .def("current_step", &Engine::current_step)
        .def("total_steps", &Engine::total_steps)
        .def("num_atoms", &Engine::num_atoms)
        .def("time_step", &Engine::time_step)
        .def("view", &Engine::view, py::return_value_policy::reference);

    py::class_<Sampler>(m, "Sampler")
        .def(py::init<GPUMDView*>())
        .def("set_callback", &Sampler::set_callback)
        .def("set_callback_interval", &Sampler::set_callback_interval)
        .def("execute", &Sampler::execute)
        .def("view", &Sampler::view, py::return_value_policy::reference);

    m.def("initialize", &initialize);
    m.def("finalize", &finalize);
    m.def("get_current_sampler", &get_current_sampler, py::return_value_policy::reference);
}
