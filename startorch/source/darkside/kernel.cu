#include "startorch/common.hpp"
#include "startorch/memory.hpp"

#include "darkside/kernel.cuh"

#include <cstdint>
#include <cstring>

namespace darkside {
template <typename T>
__global__ void fillDataGPU(T *data, uint64_t size, T value) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    data[idx] = value;
}

template <typename T> void fillDataCPU(T *data, uint64_t size, T value) {
  if (size == 0 || data == nullptr)
    return;
  if (value == (T)0) {
    std::memset(data, 0, size * sizeof(T));
    return;
  }
  std::fill_n(data, size, value);
}

template <typename T>
__global__ void fillIncreasedDataGPU(T *data, uint64_t size, T start, T step) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    data[idx] = start + (static_cast<T>(idx) * step);
}

template <typename T>
void fillIncreasedDataCPU(T *data, uint64_t size, T start, T step) {
  if (size == 0 || data == nullptr)
    return;
  for (uint64_t i = 0; i < size; i++)
    data[i] = start + (static_cast<T>(i) * step);
}

template <typename T>
__global__ void fillDecreasedDataGPU(T *data, uint64_t size, T start, T step) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    data[idx] = start + (static_cast<T>(size - 1 - idx) * step);
}

template <typename T>
void fillDecreasedDataCPU(T *data, uint64_t size, T start, T step) {
  if (size == 0 || data == nullptr)
    return;
  for (uint64_t i = 0; i < size; i++)
    data[i] = start + (static_cast<T>(size - 1 - i) * step);
}

template <typename T>
__global__ void fillStridesGPU(T *strides, T *shape, T *order, uint64_t size) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0) {
    if (size == 0)
      return;
    T current_stride = 1;
    strides[static_cast<uint64_t>(order[size - 1])] = current_stride;
    for (uint64_t i = size - 1; i > 0; --i) {
      current_stride *= shape[static_cast<uint64_t>(order[i])];
      strides[static_cast<uint64_t>(order[i - 1])] = current_stride;
    }
  }
}

template <typename T>
void fillStridesCPU(T *strides, T *shape, T *order, uint64_t size) {
  if (size == 0)
    return;
  T current_stride = 1;
  strides[static_cast<uint64_t>(order[size - 1])] = current_stride;
  for (uint64_t i = size - 1; i > 0; --i) {
    current_stride *= shape[static_cast<uint64_t>(order[i])];
    strides[static_cast<uint64_t>(order[i - 1])] = current_stride;
  }
}

template <typename T>
void fillData(void *data, uint64_t size, T value, startorch::Arena *arena) {
  switch (arena->getDevice().getDeviceType()) {
  case startorch::DeviceType::CPU:
    fillDataCPU<T>((T *)data, size, value);
    break;
  case startorch::DeviceType::GPU:
    fillDataGPU<T><<<BLOCKS(size), THREADS>>>((T *)data, size, value);
    break;
  default:
    break;
  }
}

#define INSTANTIATE_FILL_DATA(T)                                               \
  template void fillData<T>(void *, uint64_t, T, startorch::Arena *);

INSTANTIATE_FILL_DATA(int8_t)
INSTANTIATE_FILL_DATA(int16_t)
INSTANTIATE_FILL_DATA(int32_t)
INSTANTIATE_FILL_DATA(int64_t)
INSTANTIATE_FILL_DATA(float)
INSTANTIATE_FILL_DATA(double)
INSTANTIATE_FILL_DATA(uint8_t)
INSTANTIATE_FILL_DATA(uint16_t)
INSTANTIATE_FILL_DATA(uint32_t)
INSTANTIATE_FILL_DATA(uint64_t)

#undef INSTANTIATE_FILL_DATA

template <typename T>
void fillIncreasedData(void *data, uint64_t size, T start, T step,
                       startorch::Arena *arena) {
  switch (arena->getDevice().getDeviceType()) {
  case startorch::DeviceType::CPU:
    fillIncreasedDataCPU<T>((T *)data, size, start, step);
    break;
  case startorch::DeviceType::GPU:
    fillIncreasedDataGPU<T>
        <<<BLOCKS(size), THREADS>>>((T *)data, size, start, step);
    break;
  default:
    break;
  }
}

#define INSTANTIATE_INCREASE(T)                                                \
  template void fillIncreasedData<T>(void *, uint64_t, T, T,                   \
                                     startorch::Arena *);

INSTANTIATE_INCREASE(int8_t)
INSTANTIATE_INCREASE(int16_t)
INSTANTIATE_INCREASE(int32_t)
INSTANTIATE_INCREASE(int64_t)
INSTANTIATE_INCREASE(float)
INSTANTIATE_INCREASE(double)
INSTANTIATE_INCREASE(uint8_t)
INSTANTIATE_INCREASE(uint16_t)
INSTANTIATE_INCREASE(uint32_t)
INSTANTIATE_INCREASE(uint64_t)

#undef INSTANTIATE_INCREASE

template <typename T>
void fillDecreasedData(void *data, uint64_t size, T start, T step,
                       startorch::Arena *arena) {
  switch (arena->getDevice().getDeviceType()) {
  case startorch::DeviceType::CPU:
    fillDecreasedDataCPU<T>((T *)data, size, start, step);
    break;
  case startorch::DeviceType::GPU:
    fillDecreasedDataGPU<T>
        <<<BLOCKS(size), THREADS>>>((T *)data, size, start, step);
    break;
  default:
    break;
  }
}

#define INSTANTIATE_DECREASE(T)                                                \
  template void fillDecreasedData<T>(void *, uint64_t, T, T,                   \
                                     startorch::Arena *);

INSTANTIATE_DECREASE(int8_t)
INSTANTIATE_DECREASE(int16_t)
INSTANTIATE_DECREASE(int32_t)
INSTANTIATE_DECREASE(int64_t)
INSTANTIATE_DECREASE(float)
INSTANTIATE_DECREASE(double)
INSTANTIATE_DECREASE(uint8_t)
INSTANTIATE_DECREASE(uint16_t)
INSTANTIATE_DECREASE(uint32_t)
INSTANTIATE_DECREASE(uint64_t)

#undef INSTANTIATE_DECREASE

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

#define INSTANTIATE_ORDERED(T)                                                 \
  template void fillOrderedData<T>(void *, uint64_t, T, T,                     \
                                   startorch::OrderType, startorch::Arena *);

INSTANTIATE_ORDERED(int8_t)
INSTANTIATE_ORDERED(int16_t)
INSTANTIATE_ORDERED(int32_t)
INSTANTIATE_ORDERED(int64_t)
INSTANTIATE_ORDERED(float)
INSTANTIATE_ORDERED(double)
INSTANTIATE_ORDERED(uint8_t)
INSTANTIATE_ORDERED(uint16_t)
INSTANTIATE_ORDERED(uint32_t)
INSTANTIATE_ORDERED(uint64_t)

#undef INSTANTIATE_ORDERED

template <typename T>
void fillStrides(void *strides, void *shape, void *order, uint64_t size,
                 startorch::Arena *arena) {
  switch (arena->getDevice().getDeviceType()) {
  case startorch::DeviceType::CPU:
    fillStridesCPU<T>((T *)strides, (T *)shape, (T *)order, size);
    break;
  case startorch::DeviceType::GPU:
    fillStridesGPU<T><<<1, 1>>>((T *)strides, (T *)shape, (T *)order, size);
    break;
  default:
    break;
  }
}

#define INSTANTIATE_STRIDES(T)                                                 \
  template void fillStrides<T>(void *, void *, void *, uint64_t,               \
                               startorch::Arena *);

INSTANTIATE_STRIDES(int8_t)
INSTANTIATE_STRIDES(int16_t)
INSTANTIATE_STRIDES(int32_t)
INSTANTIATE_STRIDES(int64_t)
INSTANTIATE_STRIDES(uint8_t)
INSTANTIATE_STRIDES(uint16_t)
INSTANTIATE_STRIDES(uint32_t)
INSTANTIATE_STRIDES(uint64_t)
INSTANTIATE_STRIDES(float)
INSTANTIATE_STRIDES(double)

#undef INSTANTIATE_STRIDES

template <typename cpp_type, typename new_cpp_type>
__global__ void convertDataTypeGPU(const cpp_type *data, new_cpp_type *new_data,
                                   uint64_t size) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    new_data[idx] = static_cast<new_cpp_type>(data[idx]);
  }
}

template <typename cpp_type, typename new_cpp_type>
void convertDataTypeCPU(const cpp_type *data, new_cpp_type *new_data,
                        uint64_t size) {
  for (uint64_t i = 0; i < size; ++i) {
    new_data[i] = static_cast<new_cpp_type>(data[i]);
  }
}

template <typename cpp_type, typename new_cpp_type>
void convertDataType(const void *data, void *new_data, uint64_t size,
                     startorch::Arena *arena) {
  if (arena->getDevice().getDeviceType() == startorch::DeviceType::CPU) {
    convertDataTypeCPU<cpp_type, new_cpp_type>((const cpp_type *)data,
                                               (new_cpp_type *)new_data, size);
  } else {
    convertDataTypeGPU<cpp_type, new_cpp_type><<<BLOCKS(size), THREADS>>>(
        (const cpp_type *)data, (new_cpp_type *)new_data, size);
  }
}

#define INSTANTIATE_SWITCH_TARGET(cpp_type, new_cpp_type)                      \
  template void convertDataType<cpp_type, new_cpp_type>(                       \
      const void *, void *, uint64_t, startorch::Arena *);

#define INSTANTIATE_SWITCH_ALL(cpp_type)                                       \
  INSTANTIATE_SWITCH_TARGET(cpp_type, int8_t)                                  \
  INSTANTIATE_SWITCH_TARGET(cpp_type, int16_t)                                 \
  INSTANTIATE_SWITCH_TARGET(cpp_type, int32_t)                                 \
  INSTANTIATE_SWITCH_TARGET(cpp_type, int64_t)                                 \
  INSTANTIATE_SWITCH_TARGET(cpp_type, uint8_t)                                 \
  INSTANTIATE_SWITCH_TARGET(cpp_type, uint16_t)                                \
  INSTANTIATE_SWITCH_TARGET(cpp_type, uint32_t)                                \
  INSTANTIATE_SWITCH_TARGET(cpp_type, uint64_t)                                \
  INSTANTIATE_SWITCH_TARGET(cpp_type, float)                                   \
  INSTANTIATE_SWITCH_TARGET(cpp_type, double)

INSTANTIATE_SWITCH_ALL(int8_t)
INSTANTIATE_SWITCH_ALL(int16_t)
INSTANTIATE_SWITCH_ALL(int32_t)
INSTANTIATE_SWITCH_ALL(int64_t)
INSTANTIATE_SWITCH_ALL(uint8_t)
INSTANTIATE_SWITCH_ALL(uint16_t)
INSTANTIATE_SWITCH_ALL(uint32_t)
INSTANTIATE_SWITCH_ALL(uint64_t)
INSTANTIATE_SWITCH_ALL(float)
INSTANTIATE_SWITCH_ALL(double)
} // namespace darkside
