#include "startorch/common.hpp"
#include "startorch/device.hpp"

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

template <typename T, typename N> __global__ void cast_data_gpu(T *destination, const N *source, uint64_t size) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    destination[idx] = static_cast<T>(source[idx]);
}

template <typename T, typename N> void cast_data_cpu(T *destination, const N *source, uint64_t size) {
  for (uint64_t i = 0; i < size; ++i)
    destination[i] = static_cast<T>(source[i]);
}

template <typename T> void fillData(T *data, uint64_t size, T value, startorch::Device *device) {
  switch (device->getDeviceType()) {
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

template <typename T> void fillIncreasedData(T *data, uint64_t size, T start, T step, startorch::Device *device) {
  switch (device->getDeviceType()) {
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

template <typename T> void fillDecreasedData(T *data, uint64_t size, T start, T step, startorch::Device *device) {
  switch (device->getDeviceType()) {
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

template <typename T, typename N>
void castData(T *destination, const N *source, uint64_t size, startorch::Device *destination_device, startorch::Device *source_device) {
  auto dst_device = destination_device->getDeviceType();
  auto src_device = source_device->getDeviceType();

  if (dst_device == startorch::DeviceType::CPU && src_device == startorch::DeviceType::CPU) {
    cast_data_cpu<T, N>(destination, source, size);
    return;
  }

  if (dst_device == startorch::DeviceType::GPU && src_device == startorch::DeviceType::GPU) {
    cast_data_gpu<T, N><<<BLOCKS(size), THREADS>>>(destination, source, size);
    cudaDeviceSynchronize();
    return;
  }

  if (dst_device == startorch::DeviceType::GPU && src_device == startorch::DeviceType::CPU) {
    T *staging = static_cast<T *>(startorch::AMD5625U.makeData(size * sizeof(T)));

    if (staging == nullptr)
      return;

    cast_data_cpu<T, N>(staging, source, size);

    startorch::DevicePair(destination_device, &startorch::AMD5625U).copyData(destination, staging, size * sizeof(T));

    startorch::AMD5625U.freeData(size * sizeof(T));

    return;
  }

  if (dst_device == startorch::DeviceType::CPU && src_device == startorch::DeviceType::GPU) {
    N *staging = static_cast<N *>(startorch::AMD5625U.makeData(size * sizeof(N)));

    if (staging == nullptr)
      return;

    startorch::DevicePair(&startorch::AMD5625U, source_device).copyData(staging, source, size * sizeof(N));

    cast_data_cpu<T, N>(destination, staging, size);

    startorch::AMD5625U.freeData(size * sizeof(N));

    return;
  }
}

#define INSTANTIATE(macro)                                                                                                                                     \
  macro(float) macro(double) macro(int8_t) macro(int16_t) macro(int32_t) macro(int64_t) macro(uint8_t) macro(uint16_t) macro(uint32_t) macro(uint64_t)

#define INSTANTIATE_FILL(T) template void fillData<T>(T *, uint64_t, T, startorch::Device *);
INSTANTIATE(INSTANTIATE_FILL)
#undef INSTANTIATE_FILL

#define INSTANTIATE_INCREASED(T) template void fillIncreasedData<T>(T *, uint64_t, T, T, startorch::Device *);
INSTANTIATE(INSTANTIATE_INCREASED)
#undef INSTANTIATE_INCREASED

#define INSTANTIATE_DECREASED(T) template void fillDecreasedData<T>(T *, uint64_t, T, T, startorch::Device *);
INSTANTIATE(INSTANTIATE_DECREASED)
#undef INSTANTIATE_DECREASED

#define INSTANTIATE_CAST_INNER(T, N) template void castData<T, N>(T *, const N *, uint64_t, startorch::Device *, startorch::Device *);
#define INSTANTIATE_CAST(T)                                                                                                                                    \
  INSTANTIATE_CAST_INNER(T, float)                                                                                                                             \
  INSTANTIATE_CAST_INNER(T, double)                                                                                                                            \
  INSTANTIATE_CAST_INNER(T, int8_t)                                                                                                                            \
  INSTANTIATE_CAST_INNER(T, int16_t)                                                                                                                           \
  INSTANTIATE_CAST_INNER(T, int32_t)                                                                                                                           \
  INSTANTIATE_CAST_INNER(T, int64_t)                                                                                                                           \
  INSTANTIATE_CAST_INNER(T, uint8_t)                                                                                                                           \
  INSTANTIATE_CAST_INNER(T, uint16_t)                                                                                                                          \
  INSTANTIATE_CAST_INNER(T, uint32_t)                                                                                                                          \
  INSTANTIATE_CAST_INNER(T, uint64_t)
INSTANTIATE(INSTANTIATE_CAST)
#undef INSTANTIATE_CAST
#undef INSTANTIATE_CAST_INNER
#undef INSTANTIATE

} // namespace darkside
