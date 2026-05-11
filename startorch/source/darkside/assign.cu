#include "startorch/common.hpp"
#include "startorch/memory.hpp"

#include "darkside/assign.cuh"

#include <algorithm>
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
__global__ void fillIncreaseDataGPU(T *data, uint64_t size) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    data[idx] = (T)idx;
}

template <typename T> void fillIncreaseDataCPU(T *data, uint64_t size) {
  if (size == 0 || data == nullptr)
    return;
  for (uint64_t i = 0; i < size; i++)
    data[i] = (T)i;
}

template <typename T>
__global__ void fillDecreaseDataGPU(T *data, uint64_t size) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    data[idx] = (T)(size - 1 - idx);
}

template <typename T> void fillDecreaseDataCPU(T *data, uint64_t size) {
  if (size == 0 || data == nullptr)
    return;
  for (uint64_t i = 0; i < size; i++)
    data[i] = (T)(size - 1 - i);
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

template void fillData<int8_t>(void *, uint64_t, int8_t, startorch::Arena *);
template void fillData<int16_t>(void *, uint64_t, int16_t, startorch::Arena *);
template void fillData<int32_t>(void *, uint64_t, int32_t, startorch::Arena *);
template void fillData<int64_t>(void *, uint64_t, int64_t, startorch::Arena *);
template void fillData<float>(void *, uint64_t, float, startorch::Arena *);
template void fillData<double>(void *, uint64_t, double, startorch::Arena *);
template void fillData<uint8_t>(void *, uint64_t, uint8_t, startorch::Arena *);
template void fillData<uint16_t>(void *, uint64_t, uint16_t,
                                 startorch::Arena *);
template void fillData<uint32_t>(void *, uint64_t, uint32_t,
                                 startorch::Arena *);
template void fillData<uint64_t>(void *, uint64_t, uint64_t,
                                 startorch::Arena *);

template <typename T>
void fillIncreaseData(void *data, uint64_t size, startorch::Arena *arena) {
  switch (arena->getDevice().getDeviceType()) {
  case startorch::DeviceType::CPU:
    fillIncreaseDataCPU<T>((T *)data, size);
    break;

  case startorch::DeviceType::GPU:
    fillIncreaseDataGPU<T><<<BLOCKS(size), THREADS>>>((T *)data, size);
    break;

  default:
    break;
  }
}

template void fillIncreaseData<int8_t>(void *, uint64_t, startorch::Arena *);
template void fillIncreaseData<int16_t>(void *, uint64_t, startorch::Arena *);
template void fillIncreaseData<int32_t>(void *, uint64_t, startorch::Arena *);
template void fillIncreaseData<int64_t>(void *, uint64_t, startorch::Arena *);
template void fillIncreaseData<float>(void *, uint64_t, startorch::Arena *);
template void fillIncreaseData<double>(void *, uint64_t, startorch::Arena *);
template void fillIncreaseData<uint8_t>(void *, uint64_t, startorch::Arena *);
template void fillIncreaseData<uint16_t>(void *, uint64_t, startorch::Arena *);
template void fillIncreaseData<uint32_t>(void *, uint64_t, startorch::Arena *);
template void fillIncreaseData<uint64_t>(void *, uint64_t, startorch::Arena *);

template <typename T>
void fillDecreaseData(void *data, uint64_t size, startorch::Arena *arena) {
  switch (arena->getDevice().getDeviceType()) {
  case startorch::DeviceType::CPU:
    fillDecreaseDataCPU<T>((T *)data, size);
    break;

  case startorch::DeviceType::GPU:
    fillDecreaseDataGPU<T><<<BLOCKS(size), THREADS>>>((T *)data, size);
    break;

  default:
    break;
  }
}

template void fillDecreaseData<int8_t>(void *, uint64_t, startorch::Arena *);
template void fillDecreaseData<int16_t>(void *, uint64_t, startorch::Arena *);
template void fillDecreaseData<int32_t>(void *, uint64_t, startorch::Arena *);
template void fillDecreaseData<int64_t>(void *, uint64_t, startorch::Arena *);
template void fillDecreaseData<float>(void *, uint64_t, startorch::Arena *);
template void fillDecreaseData<double>(void *, uint64_t, startorch::Arena *);
template void fillDecreaseData<uint8_t>(void *, uint64_t, startorch::Arena *);
template void fillDecreaseData<uint16_t>(void *, uint64_t, startorch::Arena *);
template void fillDecreaseData<uint32_t>(void *, uint64_t, startorch::Arena *);
template void fillDecreaseData<uint64_t>(void *, uint64_t, startorch::Arena *);

template <typename T>
void fillOrderedData(void *data, uint64_t size, startorch::OrderType order_type,
                     startorch::Arena *arena) {
  switch (order_type) {
  case startorch::OrderType::ROW_MAJOR:
    fillIncreaseData<T>(data, size, arena);
    break;
  case startorch::OrderType::COLUMN_MAJOR:
    fillDecreaseData<T>(data, size, arena);
    break;
  }
}

template void fillOrderedData<int8_t>(void *, uint64_t, startorch::OrderType,
                                      startorch::Arena *);
template void fillOrderedData<int16_t>(void *, uint64_t, startorch::OrderType,
                                       startorch::Arena *);
template void fillOrderedData<int32_t>(void *, uint64_t, startorch::OrderType,
                                       startorch::Arena *);
template void fillOrderedData<int64_t>(void *, uint64_t, startorch::OrderType,
                                       startorch::Arena *);
template void fillOrderedData<float>(void *, uint64_t, startorch::OrderType,
                                     startorch::Arena *);
template void fillOrderedData<double>(void *, uint64_t, startorch::OrderType,
                                      startorch::Arena *);
template void fillOrderedData<uint8_t>(void *, uint64_t, startorch::OrderType,
                                       startorch::Arena *);
template void fillOrderedData<uint16_t>(void *, uint64_t, startorch::OrderType,
                                        startorch::Arena *);
template void fillOrderedData<uint32_t>(void *, uint64_t, startorch::OrderType,
                                        startorch::Arena *);
template void fillOrderedData<uint64_t>(void *, uint64_t, startorch::OrderType,
                                        startorch::Arena *);

} // namespace darkside
