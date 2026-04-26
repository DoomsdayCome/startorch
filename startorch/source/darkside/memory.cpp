#include "darkside/memory.hpp"
#include "startorch/common.hpp"
#include "startorch/device.hpp"

#include <cstdint>
#include <cstring>
#include <new>

#include <cuda_runtime.h>

namespace darkside {
void *makeMemory(uint64_t size, const startorch::Device &device) {
  void *pointer = nullptr;

  if (device.getMemoryType() == startorch::MemoryType::PINNED) {
    if (cudaMallocHost(&pointer, size) != cudaSuccess)
      return nullptr;
    return pointer;
  }

  if (device.getMemoryType() == startorch::MemoryType::UNIFIED) {
    if (cudaMallocManaged(&pointer, size) != cudaSuccess)
      return nullptr;
    return pointer;
  }

  switch (device.getDeviceType()) {
  case startorch::DeviceType::CPU:
    pointer = new (std::nothrow) uint8_t[size];
    break;

  case startorch::DeviceType::GPU:
    if (cudaMalloc(&pointer, size) != cudaSuccess)
      pointer = nullptr;
    break;

  default:
    break;
  }

  return pointer;
}

void freeMemory(void *pointer, const startorch::Device &device) {
  if (pointer == nullptr)
    return;

  if (device.getMemoryType() == startorch::MemoryType::PINNED) {
    cudaFreeHost(pointer);
    return;
  }

  if (device.getMemoryType() == startorch::MemoryType::UNIFIED) {
    cudaFree(pointer);
    return;
  }

  switch (device.getDeviceType()) {
  case startorch::DeviceType::CPU:
    delete[] static_cast<uint8_t *>(pointer);
    break;

  case startorch::DeviceType::GPU:
    cudaFree(pointer);
    break;
  
  default:
    break;
  }
}

void copyMemory(void *destination, void *source, uint64_t size,
                const startorch::DevicePair &device_pair) {
  if (destination == nullptr || source == nullptr || size == 0)
    return;

  startorch::DeviceType first_device_type =
      device_pair.getFirstDevice().getDeviceType();
  startorch::DeviceType second_device_type =
      device_pair.getSecondDevice().getDeviceType();

  if (first_device_type == startorch::DeviceType::CPU &&
      second_device_type == startorch::DeviceType::CPU)
    memcpy(destination, source, size);
  else
    cudaMemcpy(destination, source, size, cudaMemcpyDefault);
}
} // namespace darkside
