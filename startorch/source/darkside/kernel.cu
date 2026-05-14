#include "startorch/common.hpp"
#include "startorch/device.hpp"
#include "startorch/memory.hpp"

#include "darkside/kernel.cuh"

#include <cstdint>
#include <cstring>
#include <sys/types.h>

namespace darkside {
template <typename T> __global__ void fill_data_gpu(T *data, uint64_t size, T value) {
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

template <typename T> __global__ void fill_increased_data_gpu(T *data, uint64_t size, T start, T step) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    data[idx] = start + (static_cast<T>(idx) * step);
}

template <typename T> void fill_increased_data_cpu(T *data, uint64_t size, T start, T step) {
  if (size == 0 || data == nullptr)
    return;
  for (uint64_t i = 0; i < size; i++)
    data[i] = start + (static_cast<T>(i) * step);
}

template <typename T> __global__ void fill_decreased_data_gpu(T *data, uint64_t size, T start, T step) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    data[idx] = start + (static_cast<T>(size - 1 - idx) * step);
}

template <typename T> void fill_decreased_data_cpu(T *data, uint64_t size, T start, T step) {
  if (size == 0 || data == nullptr)
    return;
  for (uint64_t i = 0; i < size; i++)
    data[i] = start + (static_cast<T>(size - 1 - i) * step);
}

template <typename T> __global__ void fill_strides_gpu(T *strides, T *shape, T *order, uint64_t size) {
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

template <typename T> void fill_strides_cpu(T *strides, T *shape, T *order, uint64_t size) {
  if (size == 0)
    return;
  T current_stride = 1;
  strides[static_cast<uint64_t>(order[size - 1])] = current_stride;
  for (uint64_t i = size - 1; i > 0; --i) {
    current_stride *= shape[static_cast<uint64_t>(order[i])];
    strides[static_cast<uint64_t>(order[i - 1])] = current_stride;
  }
}

template <typename cpp_type, typename new_cpp_type> __global__ void convert_data_type_gpu(const cpp_type *data, new_cpp_type *new_data, uint64_t size) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    new_data[idx] = static_cast<new_cpp_type>(data[idx]);
}

template <typename cpp_type, typename new_cpp_type> void convert_data_type_cpu(const cpp_type *data, new_cpp_type *new_data, uint64_t size) {
  for (uint64_t i = 0; i < size; ++i)
    new_data[i] = static_cast<new_cpp_type>(data[i]);
}

template <typename T> __global__ void fill_storage_index_gpu_kernel(uint64_t *index, T *indices, T *strides, T *offsets, uint64_t rank) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint64_t storage_idx = 0;
    for (uint64_t j = 0; j < rank; ++j) {
      storage_idx += static_cast<uint64_t>(indices[j]) * static_cast<uint64_t>(strides[j]);
    }
    *index = storage_idx + static_cast<uint64_t>(offsets[0]);
  }
}

template <typename T> void fill_storage_index_cpu(uint64_t *index, T *indices, T *strides, T *offsets, uint64_t rank) {
  uint64_t storage_idx = 0;
  for (uint64_t j = 0; j < rank; ++j) {
    storage_idx += static_cast<uint64_t>(indices[j]) * static_cast<uint64_t>(strides[j]);
  }
  *index = storage_idx + static_cast<uint64_t>(offsets[0]);
}

template <typename T> void fillData(void *data, uint64_t size, T value, startorch::Arena *arena) {
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

template <typename T> void fillIncreasedData(void *data, uint64_t size, T start, T step, startorch::Arena *arena) {
  switch (arena->getDevice().getDeviceType()) {
  case startorch::DeviceType::CPU:
    fill_increased_data_cpu<T>((T *)data, size, start, step);
    break;
  case startorch::DeviceType::GPU:
    fill_increased_data_gpu<T><<<BLOCKS(size), THREADS>>>((T *)data, size, start, step);
    break;
  default:
    break;
  }
}

template <typename T> void fillDecreasedData(void *data, uint64_t size, T start, T step, startorch::Arena *arena) {
  switch (arena->getDevice().getDeviceType()) {
  case startorch::DeviceType::CPU:
    fill_decreased_data_cpu<T>((T *)data, size, start, step);
    break;
  case startorch::DeviceType::GPU:
    fill_decreased_data_gpu<T><<<BLOCKS(size), THREADS>>>((T *)data, size, start, step);
    break;
  default:
    break;
  }
}

template <typename T> void fillOrderedData(void *data, uint64_t size, T start, T step, startorch::OrderType order_type, startorch::Arena *arena) {
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

template <typename T> void fillStrides(void *strides, void *shape, void *order, uint64_t size, startorch::Arena *arena) {
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

template <typename T> void fill_storage_index_cpu(uint64_t *index, T *indices, T *strides, T *offsets, uint64_t rank, uint64_t size) {
  for (uint64_t i = 0; i < size; ++i) {
    uint64_t linear_idx = i;
    uint64_t storage_offset = 0;

    for (int64_t j = rank - 1; j >= 0; --j) {
      uint64_t coord = linear_idx % static_cast<uint64_t>(indices[j]);
      linear_idx /= static_cast<uint64_t>(indices[j]);
      storage_offset += coord * static_cast<uint64_t>(strides[j]);
    }

    index[i] = storage_offset + static_cast<uint64_t>(offsets[0]);
  }
}

template <typename cpp_type, typename new_cpp_type> void convertDataType(const void *data, void *new_data, uint64_t size, startorch::Arena *arena) {
  if (arena->getDevice().getDeviceType() == startorch::DeviceType::CPU) {
    convert_data_type_cpu<cpp_type, new_cpp_type>((const cpp_type *)data, (new_cpp_type *)new_data, size);
  } else {
    convert_data_type_gpu<cpp_type, new_cpp_type><<<BLOCKS(size), THREADS>>>((const cpp_type *)data, (new_cpp_type *)new_data, size);
  }
}

template <typename T> void fillStorageIndex(uint64_t *index, void *indices, void *strides, void *offsets, uint64_t size, startorch::Arena *arena) {
  T *d_indices = static_cast<T *>(indices);
  T *d_strides = static_cast<T *>(strides);
  T *d_offsets = static_cast<T *>(offsets);

  if (arena->getDevice().getDeviceType() == startorch::DeviceType::CPU) {
    fill_storage_index_cpu<T>(index, d_indices, d_strides, d_offsets, size);
  } else {
    uint64_t *index_gpu = (uint64_t *)arena->makeData(sizeof(uint64_t));

    fill_storage_index_gpu_kernel<T><<<1, 1>>>(index_gpu, d_indices, d_strides, d_offsets, size);

    startorch::Arena::copyData(index, index_gpu, sizeof(uint64_t), startorch::DevicePair(arena->getDevice(), startorch::Device()));

    arena->freeData(sizeof(uint64_t));
  }
}

#define INSTANTIATE(macro)                                                                                                                                     \
  macro(int8_t) macro(int16_t) macro(int32_t) macro(int64_t) macro(uint8_t) macro(uint16_t) macro(uint32_t) macro(uint64_t) macro(float) macro(double)

#define INST_FILL_DATA(T) template void fillData<T>(void *, uint64_t, T, startorch::Arena *);
INSTANTIATE(INST_FILL_DATA)
#undef INST_FILL_DATA

#define INST_INCREASED(T) template void fillIncreasedData<T>(void *, uint64_t, T, T, startorch::Arena *);
INSTANTIATE(INST_INCREASED)
#undef INST_INCREASED

#define INST_DECREASED(T) template void fillDecreasedData<T>(void *, uint64_t, T, T, startorch::Arena *);
INSTANTIATE(INST_DECREASED)
#undef INST_DECREASED

#define INST_ORDERED(T) template void fillOrderedData<T>(void *, uint64_t, T, T, startorch::OrderType, startorch::Arena *);
INSTANTIATE(INST_ORDERED)
#undef INST_ORDERED

#define INST_STRIDES(T) template void fillStrides<T>(void *, void *, void *, uint64_t, startorch::Arena *);
INSTANTIATE(INST_STRIDES)
#undef INST_STRIDES

#define CONVERT_TARGET(cpp_type, new_cpp_type) template void convertDataType<cpp_type, new_cpp_type>(const void *, void *, uint64_t, startorch::Arena *);

#define INST_INDEX(T) template void fillStorageIndex<T>(uint64_t *, void *, void *, void *, uint64_t, startorch::Arena *);

INSTANTIATE(INST_INDEX)
#undef INST_INDEX

#define CONVERT_ALL(cpp_type)                                                                                                                                  \
  CONVERT_TARGET(cpp_type, int8_t)                                                                                                                             \
  CONVERT_TARGET(cpp_type, int16_t)                                                                                                                            \
  CONVERT_TARGET(cpp_type, int32_t)                                                                                                                            \
  CONVERT_TARGET(cpp_type, int64_t)                                                                                                                            \
  CONVERT_TARGET(cpp_type, uint8_t)                                                                                                                            \
  CONVERT_TARGET(cpp_type, uint16_t)                                                                                                                           \
  CONVERT_TARGET(cpp_type, uint32_t)                                                                                                                           \
  CONVERT_TARGET(cpp_type, uint64_t)                                                                                                                           \
  CONVERT_TARGET(cpp_type, float)                                                                                                                              \
  CONVERT_TARGET(cpp_type, double)

INSTANTIATE(CONVERT_ALL)

#undef CONVERT_ALL
#undef CONVERT_TARGET
#undef INSTANTIATE

} // namespace darkside
