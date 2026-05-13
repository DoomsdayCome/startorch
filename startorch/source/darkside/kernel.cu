#include "startorch/common.hpp"
#include "startorch/memory.hpp"

#include "darkside/kernel.cuh"

#include <cstdint>
#include <cstring>

namespace darkside {
template <typename T>
__global__ void fill_data_gpu(T *data, uint64_t size, T value) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    data[idx] = value;
}

template <typename T> void fill_data_cpu(T *data, uint64_t size, T value) {
  if (size == 0 || data == nullptr)
    return;
  if (value == (T)0) {
    std::memset(data, 0, size * sizeof(T));
    return;
  }
  std::fill_n(data, size, value);
}

template <typename T>
__global__ void fill_increased_data_gpu(T *data, uint64_t size, T start,
                                        T step) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    data[idx] = start + (static_cast<T>(idx) * step);
}

template <typename T>
void fill_increased_data_cpu(T *data, uint64_t size, T start, T step) {
  if (size == 0 || data == nullptr)
    return;
  for (uint64_t i = 0; i < size; i++)
    data[i] = start + (static_cast<T>(i) * step);
}

template <typename T>
__global__ void fill_decreased_data_gpu(T *data, uint64_t size, T start,
                                        T step) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    data[idx] = start + (static_cast<T>(size - 1 - idx) * step);
}

template <typename T>
void fill_decreased_data_cpu(T *data, uint64_t size, T start, T step) {
  if (size == 0 || data == nullptr)
    return;
  for (uint64_t i = 0; i < size; i++)
    data[i] = start + (static_cast<T>(size - 1 - i) * step);
}

template <typename T>
__global__ void fill_strides_gpu(T *strides, T *shape, T *order,
                                 uint64_t size) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx != 0 || size == 0)
    return;
  T current_stride = 1;
  strides[static_cast<uint64_t>(order[size - 1])] = current_stride;
  for (uint64_t i = size - 1; i > 0; --i) {
    current_stride *= shape[static_cast<uint64_t>(order[i])];
    strides[static_cast<uint64_t>(order[i - 1])] = current_stride;
  }
}

template <typename T>
void fill_strides_cpu(T *strides, T *shape, T *order, uint64_t size) {
  if (size == 0)
    return;
  T current_stride = 1;
  strides[static_cast<uint64_t>(order[size - 1])] = current_stride;
  for (uint64_t i = size - 1; i > 0; --i) {
    current_stride *= shape[static_cast<uint64_t>(order[i])];
    strides[static_cast<uint64_t>(order[i - 1])] = current_stride;
  }
}

template <typename cpp_type, typename new_cpp_type>
__global__ void convert_data_type_gpu(const cpp_type *data,
                                      new_cpp_type *new_data, uint64_t size) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    new_data[idx] = static_cast<new_cpp_type>(data[idx]);
}

template <typename cpp_type, typename new_cpp_type>
void convert_data_type_cpu(const cpp_type *data, new_cpp_type *new_data,
                           uint64_t size) {
  for (uint64_t i = 0; i < size; ++i)
    new_data[i] = static_cast<new_cpp_type>(data[i]);
}

template <typename T>
void fillData(void *data, uint64_t size, T value, startorch::Arena *arena) {
  switch (arena->getDevice().getDeviceType()) {
  case startorch::DeviceType::CPU:
    fill_data_cpu<T>((T *)data, size, value);
    break;
  case startorch::DeviceType::GPU:
    fill_data_gpu<T><<<BLOCKS(size), THREADS>>>((T *)data, size, value);
    break;
  default:
    break;
  }
}

template <typename T>
void fillIncreasedData(void *data, uint64_t size, T start, T step,
                       startorch::Arena *arena) {
  switch (arena->getDevice().getDeviceType()) {
  case startorch::DeviceType::CPU:
    fill_increased_data_cpu<T>((T *)data, size, start, step);
    break;
  case startorch::DeviceType::GPU:
    fill_increased_data_gpu<T>
        <<<BLOCKS(size), THREADS>>>((T *)data, size, start, step);
    break;
  default:
    break;
  }
}

template <typename T>
void fillDecreasedData(void *data, uint64_t size, T start, T step,
                       startorch::Arena *arena) {
  switch (arena->getDevice().getDeviceType()) {
  case startorch::DeviceType::CPU:
    fill_decreased_data_cpu<T>((T *)data, size, start, step);
    break;
  case startorch::DeviceType::GPU:
    fill_decreased_data_gpu<T>
        <<<BLOCKS(size), THREADS>>>((T *)data, size, start, step);
    break;
  default:
    break;
  }
}

template <typename T>
void fillOrderedData(void *data, uint64_t size, T start, T step,
                     startorch::OrderType order_type, startorch::Arena *arena) {
  switch (order_type) {
  case startorch::OrderType::ROW_MAJOR:
    fillIncreasedData<T>(data, size, start, step, arena);
    break;
  case startorch::OrderType::COLUMN_MAJOR:
    fillDecreasedData<T>(data, size, start, step, arena);
    break;
  default:
    break;
  }
}

template <typename T>
void fillStrides(void *strides, void *shape, void *order, uint64_t size,
                 startorch::Arena *arena) {
  switch (arena->getDevice().getDeviceType()) {
  case startorch::DeviceType::CPU:
    fill_strides_cpu<T>((T *)strides, (T *)shape, (T *)order, size);
    break;
  case startorch::DeviceType::GPU:
    fill_strides_gpu<T><<<1, 1>>>((T *)strides, (T *)shape, (T *)order, size);
    break;
  default:
    break;
  }
}

template <typename cpp_type, typename new_cpp_type>
void convertDataType(const void *data, void *new_data, uint64_t size,
                     startorch::Arena *arena) {
  if (arena->getDevice().getDeviceType() == startorch::DeviceType::CPU) {
    convert_data_type_cpu<cpp_type, new_cpp_type>(
        (const cpp_type *)data, (new_cpp_type *)new_data, size);
  } else {
    convert_data_type_gpu<cpp_type, new_cpp_type><<<BLOCKS(size), THREADS>>>(
        (const cpp_type *)data, (new_cpp_type *)new_data, size);
  }
}

#define INSTANTIATE(macro)                                                     \
  macro(int8_t) macro(int16_t) macro(int32_t) macro(int64_t) macro(uint8_t)    \
      macro(uint16_t) macro(uint32_t) macro(uint64_t) macro(float)             \
          macro(double)

#define INST_FILL_DATA(T)                                                      \
  template void fillData<T>(void *, uint64_t, T, startorch::Arena *);
INSTANTIATE(INST_FILL_DATA)
#undef INST_FILL_DATA

#define INST_INCREASED(T)                                                      \
  template void fillIncreasedData<T>(void *, uint64_t, T, T,                   \
                                     startorch::Arena *);
INSTANTIATE(INST_INCREASED)
#undef INST_INCREASED

#define INST_DECREASED(T)                                                      \
  template void fillDecreasedData<T>(void *, uint64_t, T, T,                   \
                                     startorch::Arena *);
INSTANTIATE(INST_DECREASED)
#undef INST_DECREASED

#define INST_ORDERED(T)                                                        \
  template void fillOrderedData<T>(void *, uint64_t, T, T,                     \
                                   startorch::OrderType, startorch::Arena *);
INSTANTIATE(INST_ORDERED)
#undef INST_ORDERED

#define INST_STRIDES(T)                                                        \
  template void fillStrides<T>(void *, void *, void *, uint64_t,               \
                               startorch::Arena *);
INSTANTIATE(INST_STRIDES)
#undef INST_STRIDES

#define CONVERT_TARGET(cpp_type, new_cpp_type)                                 \
  template void convertDataType<cpp_type, new_cpp_type>(                       \
      const void *, void *, uint64_t, startorch::Arena *);

#define CONVERT_ALL(cpp_type)                                                  \
  CONVERT_TARGET(cpp_type, int8_t)                                             \
  CONVERT_TARGET(cpp_type, int16_t)                                            \
  CONVERT_TARGET(cpp_type, int32_t)                                            \
  CONVERT_TARGET(cpp_type, int64_t)                                            \
  CONVERT_TARGET(cpp_type, uint8_t)                                            \
  CONVERT_TARGET(cpp_type, uint16_t)                                           \
  CONVERT_TARGET(cpp_type, uint32_t)                                           \
  CONVERT_TARGET(cpp_type, uint64_t)                                           \
  CONVERT_TARGET(cpp_type, float)                                              \
  CONVERT_TARGET(cpp_type, double)

INSTANTIATE(CONVERT_ALL)

#undef CONVERT_ALL
#undef CONVERT_TARGET
#undef INSTANTIATE

} // namespace darkside
