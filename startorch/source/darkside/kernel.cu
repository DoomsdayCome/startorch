#include "startorch/common.hpp"
#include "startorch/device.hpp"
#include "startorch/memory.hpp"

#include "darkside/kernel.cuh"

#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>

namespace darkside {
template <typename T> __global__ void fill_data_gpu(T *data, uint64_t size, T value) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size)
    data[idx] = value;
}

template <typename T> void fill_data_cpu(T *data, uint64_t size, T value) {
  if (value == (T)0) {
    std::memset(data, 0, size * sizeof(T));
    return;
  }

  std::fill_n(data, size, value);
}

template <typename T> __global__ void fill_increased_data_gpu(T *data, uint64_t size, T start, T step) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size)
    data[idx] = start + (static_cast<T>(idx) * step);
}

template <typename T> void fill_increased_data_cpu(T *data, uint64_t size, T start, T step) {
  for (uint64_t i = 0; i < size; i++)
    data[i] = start + (static_cast<T>(i) * step);
}

template <typename T> __global__ void fill_decreased_data_gpu(T *data, uint64_t size, T start, T step) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size)
    data[idx] = start - (static_cast<T>(idx) * step);
}

template <typename T> void fill_decreased_data_cpu(T *data, uint64_t size, T start, T step) {
  for (uint64_t i = 0; i < size; i++)
    data[i] = start - (static_cast<T>(i) * step);
}

template <typename T> void fillData(T *data, uint64_t size, T value, startorch::Arena *arena) {
  switch (arena->getDevice().getDeviceType()) {
  case startorch::DeviceType::CPU:
    fill_data_cpu<T>(data, size, value);
    break;

  case startorch::DeviceType::GPU:
    if (value == 0) {
      cudaMemset(data, value, size * sizeof(T));
      return;
    }

    fill_data_gpu<T><<<BLOCKS(size), THREADS>>>(data, size, value);
    break;

  default:
    break;
  }
}

template <typename T> void fillIncreasedData(T *data, uint64_t size, T start, T step, startorch::Arena *arena) {
  switch (arena->getDevice().getDeviceType()) {
  case startorch::DeviceType::CPU:
    fill_increased_data_cpu<T>(data, size, start, step);
    break;

  case startorch::DeviceType::GPU:
    fill_increased_data_gpu<T><<<BLOCKS(size), THREADS>>>(data, size, start, step);
    break;

  default:
    break;
  }
}

template <typename T> void fillDecreasedData(T *data, uint64_t size, T start, T step, startorch::Arena *arena) {
  switch (arena->getDevice().getDeviceType()) {
  case startorch::DeviceType::CPU:
    fill_decreased_data_cpu<T>(data, size, start, step);
    break;

  case startorch::DeviceType::GPU:
    fill_decreased_data_gpu<T><<<BLOCKS(size), THREADS>>>(data, size, start, step);
    break;

  default:
    break;
  }
}

#define INSTANTIATE(macro)                                                                                                                                     \
  macro(int8_t) macro(int16_t) macro(int32_t) macro(int64_t) macro(float) macro(double) macro(uint8_t) macro(uint16_t) macro(uint32_t) macro(uint64_t)

#define INSTANTIATE_FILL(T) template void fillData<T>(T *, uint64_t, T, startorch::Arena *);
INSTANTIATE(INSTANTIATE_FILL)
#undef INSTANTIATE_FILL

#define INSTANTIATE_INCREASED(T) template void fillIncreasedData<T>(T *, uint64_t, T, T, startorch::Arena *);
INSTANTIATE(INSTANTIATE_INCREASED)
#undef INSTANTIATE_INCREASED

#define INSTANTIATE_DECREASED(T) template void fillDecreasedData<T>(T *, uint64_t, T, T, startorch::Arena *);
INSTANTIATE(INSTANTIATE_DECREASED)
#undef INSTANTIATE_DECREASED

#undef INSTANTIATE

} // namespace darkside
