#include "startorch/memory.hpp"
#include "startorch/common.hpp"
#include "startorch/device.hpp"
#include "startorch/format.hpp"

#include <cstdint>
#include <cstring>
#include <new>

#include <cuda_runtime.h>

namespace startorch {
void *makeData(uint64_t size, const Device &device) {
  void *pointer = nullptr;

  if (device.getMemoryType() == MemoryType::PINNED) {
    if (cudaMallocHost(&pointer, size) != cudaSuccess)
      return nullptr;
    return pointer;
  }

  if (device.getMemoryType() == MemoryType::UNIFIED) {
    if (cudaMallocManaged(&pointer, size) != cudaSuccess)
      return nullptr;
    return pointer;
  }

  switch (device.getDeviceType()) {
  case DeviceType::CPU:
    pointer = new (std::nothrow) uint8_t[size];
    break;

  case DeviceType::GPU:
    if (cudaMalloc(&pointer, size) != cudaSuccess)
      pointer = nullptr;
    break;

  default:
    break;
  }

  return pointer;
}

void freeData(void *pointer, const Device &device) {
  if (pointer == nullptr)
    return;

  if (device.getMemoryType() == MemoryType::PINNED) {
    cudaFreeHost(pointer);
    return;
  }

  if (device.getMemoryType() == MemoryType::UNIFIED) {
    cudaFree(pointer);
    return;
  }

  switch (device.getDeviceType()) {
  case DeviceType::CPU:
    delete[] static_cast<uint8_t *>(pointer);
    break;

  case DeviceType::GPU:
    cudaFree(pointer);
    break;

  default:
    break;
  }
}

void copyData(void *destination, void *source, uint64_t size,
              const DevicePair &device_pair) {
  if (destination == nullptr || source == nullptr || size == 0)
    return;

  DeviceType first_device_type = device_pair.getFirstDevice().getDeviceType();
  DeviceType second_device_type = device_pair.getSecondDevice().getDeviceType();

  if (first_device_type == DeviceType::CPU &&
      second_device_type == DeviceType::CPU)
    memcpy(destination, source, size);
  else
    cudaMemcpy(destination, source, size, cudaMemcpyDefault);
}

Storage::Storage(uint64_t size, ScalarType scalar_type, const Device &device)
    : size_(size), scalar_type_(scalar_type), device_(device) {
  if (size_ == 0)
    return;

  data_ = makeData(size_ * getScalarTypeSize(scalar_type_), device_);

  if (data_ == nullptr)
    size_ = 0;
}

Storage::Storage(const Storage &other)
    : size_(other.size_), scalar_type_(other.scalar_type_),
      device_(other.device_) {
  if (size_ == 0)
    return;

  data_ = makeData(size_ * getScalarTypeSize(scalar_type_), device_);

  if (data_ == nullptr) {
    size_ = 0;
    return;
  }

  copyData(data_, other.data_, size_ * getScalarTypeSize(scalar_type_),
           DevicePair(device_, other.device_));
}

Storage::Storage(Storage &&other) noexcept
    : data_(other.data_), size_(other.size_), scalar_type_(other.scalar_type_),
      device_(other.device_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

Storage::~Storage() {
  if (data_ != nullptr)
    freeData(data_, device_);
}

Storage &Storage::operator=(const Storage &other) {
  if (this == &other)
    return *this;

  if (other.size_ == 0) {
    if (size_ != 0)
      freeData(data_, device_);

    data_ = nullptr;
    size_ = 0;
    scalar_type_ = other.scalar_type_;
    device_ = other.device_;

    return *this;
  }

  uint64_t bytes = other.size_ * getScalarTypeSize(other.scalar_type_);
  void *new_data = makeData(bytes, other.device_);

  if (new_data == nullptr)
    return *this;

  if (size_ != 0)
    freeData(data_, device_);

  data_ = new_data;
  size_ = other.size_;
  scalar_type_ = other.scalar_type_;
  device_ = other.device_;

  copyData(data_, other.data_, bytes, DevicePair(device_, other.device_));

  return *this;
}

Storage &Storage::operator=(Storage &&other) noexcept {
  if (this == &other)
    return *this;

  if (size_ != 0)
    freeData(data_, device_);

  data_ = other.data_;
  size_ = other.size_;
  scalar_type_ = other.scalar_type_;
  device_ = other.device_;

  other.data_ = nullptr;
  other.size_ = 0;

  return *this;
}

void *Storage::getData() const { return data_; }
uint64_t Storage::getSize() const { return size_; }
ScalarType Storage::getScalarType() const { return scalar_type_; }
const Device &Storage::getDevice() const { return device_; }

void Storage::setDevice(const Device &device) {
  if (device == device_ || size_ == 0) {
    device_ = device;
    return;
  }

  uint64_t bytes = size_ * getScalarTypeSize(scalar_type_);
  void *new_data = makeData(bytes, device);

  if (new_data == nullptr)
    return;

  copyData(new_data, data_, bytes, DevicePair(device_, device));

  if (data_ != nullptr)
    freeData(data_, device_);

  data_ = new_data;
  device_ = device;
}
} // namespace startorch
