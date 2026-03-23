/*!
 *  Copyright (c) 2017 by Contributors
 * \file dlpack.h
 * \brief The common header of DLPack.
 */
#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_

#ifdef __cplusplus
#define DLPACK_EXTERN_C extern "C"
#else
#define DLPACK_EXTERN_C
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief The device type in DLDevice.
 */
typedef enum {
  /*! \brief CPU device */
  kDLCPU = 1,
  /*! \brief CUDA GPU device */
  kDLCUDA = 2,
  /*! \brief Pinned CUDA CPU memory by cudaMallocHost */
  kDLCUDAHost = 3,
  /*! \brief OpenCL devices */
  kDLOpenCL = 4,
  /*! \brief Vulkan buffer for next generation graphics */
  kDLVulkan = 7,
  /*! \brief Metal for Apple GPU */
  kDLMetal = 8,
  /*! \brief Verilog simulator buffer */
  kDLVPI = 9,
  /*! \brief ROCm GPUs for AMD GPUs */
  kDLROCM = 10,
  /*! \brief Pinned ROCm CPU memory allocated with hipMallocHost */
  kDLROCMHost = 11,
  /*! \brief Reserved extension device type,
   * used for quickly test extension device
   * The semantics can differ depending on the implementation.
   */
  kDLExtDev = 12,
  /*! \brief CUDA Managed Memory */
  kDLCUDAManaged = 13,
  /*! \brief Unified shared memory allocated on oneAPI non-partitoned devices.
   * \(https://spec.oneapi.io/level-zero/latest/core/UNIF.html\ */
  kDLOneAPI = 14,
  /*! \brief GPU support for next generation WebGPU standard.
   * (https://www.w3.org/TR/webgpu/)
   */
  kDLWebGPU = 15,
  /*! \brief Qualcomm Hexagon DSP */
  kDLHexagon = 16,
} DLDeviceType;

/*!
 * \brief A Device for Tensor and operator.
 */
typedef struct {
  /*! \brief The device type used in the device. */
  DLDeviceType device_type;
  /*! \brief The device index. */
  int device_id;
} DLDevice;

/*!
 * \brief The type code options DLDataType.
 */
typedef enum {
  /*! \brief signed integer */
  kDLInt = 0U,
  /*! \brief unsigned integer */
  kDLUInt = 1U,
  /*! \brief IEEE floating point */
  kDLFloat = 2U,
  /*! \brief Opaque handle type, reserved for testing purposes. */
  kDLOpaqueHandle = 3U,
  /*! \brief bfloat16 */
  kDLBfloat = 4U,
  /*! \brief complex number
   * (C/C++/Python layout: compact struct per complex number)
   */
  kDLComplex = 5U,
  /*! \brief boolean */
  kDLBool = 6U,
} DLDataTypeCode;

/*!
 * \brief The data type the tensor can hold.
 *
 * The data type is mainly used to indicate the type of the data content.
 *
 * The data type can be used to support more precise type dispatch
 * in programs.
 *
 * The field lanes is not used by DLPack and can be set to 1.
 */
typedef struct {
  /*! \brief Type code of base types.
   * We keep it uint8_t instead of DLDataTypeCode for minimal memory
   * footprint, but the value should be one of DLDataTypeCode enum values.
   */
  uint8_t code;
  /*! \brief Number of bits, commonly 32, 64 */
  uint8_t bits;
  /*! \brief Number of lanes in the type, used for vector types. */
  uint16_t lanes;
} DLDataType;

/*!
 * \brief Plain C Tensor object, does not hold memory.
 *
 * This data structure is designed to hold memory-managed data and not
 * provide additional memory management. For example, the data field
 * is a borrowed reference to an external data, which may be managed
 * by frameworks (e.g., frameworks that have reference counting).
 *
 * This data structure is used to pass data between different frameworks.
 * An exchange may happen in the form of borrowing or moving.
 * When moving, the exchanging party does not hold reference to the object
 * after exchange.
 */
typedef struct {
  /*! \brief The data pointer. */
  void* data;
  /*! \brief The device of the tensor. */
  DLDevice device;
  /*! \brief Number of dimensions. */
  int32_t ndim;
  /*! \brief The data type of the pointer. */
  DLDataType dtype;
  /*! \brief The shape of the tensor. */
  int64_t* shape;
  /*! \brief Strides of the tensor in each dimension.
   *  This can be NULL, indicating tensor is compact and row-majored.
   */
  int64_t* strides;
  /*! \brief The offset in bytes to the beginning pointer to data. */
  uint64_t byte_offset;
} DLTensor;

/*!
 * \brief C Tensor object, hold memory reference.
 *
 * This data structure is a reference to the underlying memory and shape.
 *
 * When the data is owned by the object, the manager_ctx holds reference
 * to the underlying framework context(e.g. for PyTorch framework, the
 * manager_ctx holds the reference to THPObject).
 *
 * \code
 *
 * // The framework object that holds the memory pointer and deleter.
 * struct MyTensorObject {
 *     DLMTensorView tensor;
 *     std::vector<int64_t> shape;
 *     std::string name;
 * };
 *
 * // Define deleter that deletes the MyTensorObject wrapper.
 * void deleter(DLManagedTensor* self) {
 *     delete static_cast<MyTensorObject*>(self->manager_ctx);
 * }
 *
 * \endcode
 */
typedef struct DLManagedTensor {
  /*! \brief DLTensor which is being memory managed. */
  DLTensor dl_tensor;
  /*! \brief the context of the original host framework of DLManagedTensor in
   *   DLManagedTensor is used to identify which framework is managing the data.
   *   Manager_ctx can be NULL.
   *   It is recommended to use a pointer to the current framework's
   *   DLManagedTensor as the manager_ctx.
   */
  void* manager_ctx;
  /*! \brief Destructor signature void (*)(void*) - this should be called
   *   to destruct manager_ctx which holds the DLManagedTensor. It can be NULL
   *   if there is no way for the caller to provide a reasonable destructor.
   *   The destructors deletes the argument self as well.
   */
  void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

#ifdef __cplusplus
}  // DLPACK_EXTERN_C
#endif
#endif  // DLPACK_DLPACK_H_
