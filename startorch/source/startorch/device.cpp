#include "startorch/device.hpp"
#include "startorch/common.hpp"

#include <cstring>
#include <new>

#include <cuda_runtime.h>

namespace startorch {
Device::Device(uint64_t bytes, MemoryType memory_type, DeviceType device_type) {
  if (bytes == 0 || memory_type == MemoryType::UNKNOWN_MEMORY || device_type == DeviceType::UNKNOWN_DEVICE)
    return;

  bytes_ = bytes;
  memory_type_ = memory_type;
  device_type_ = device_type;
  offset_ = 0;

  if (device_type_ == DeviceType::CPU) {
    if (memory_type_ == MemoryType::DEVICE || memory_type_ == MemoryType::UNIFIED)
      memory_type_ = MemoryType::HOST;
  } else if (device_type_ == DeviceType::GPU) {
    if (memory_type_ == MemoryType::HOST || memory_type_ == MemoryType::PINNED)
      memory_type_ = MemoryType::DEVICE;
  }

  switch (memory_type_) {
  case MemoryType::HOST:
    data_ = new (std::nothrow) uint8_t[bytes_];
    break;

  case MemoryType::DEVICE:
    if (cudaMalloc(&data_, bytes_) != cudaSuccess)
      data_ = nullptr;

    break;

  case MemoryType::PINNED:
    if (cudaMallocHost(&data_, bytes_) != cudaSuccess)
      data_ = nullptr;

    break;

  case MemoryType::UNIFIED:
    if (cudaMallocManaged(&data_, bytes_) != cudaSuccess)
      data_ = nullptr;

    break;

  default:
    break;
  }

  if (data_ == nullptr) {
    bytes_ = 0;
    offset_ = 0;
    memory_type_ = MemoryType::UNKNOWN_MEMORY;
    device_type_ = DeviceType::UNKNOWN_DEVICE;
  }
}

Device::~Device() {
  if (bytes_ == 0)
    return;

  switch (memory_type_) {
  case MemoryType::HOST:
    delete[] static_cast<uint8_t *>(data_);

    break;

  case MemoryType::PINNED:
    cudaFreeHost(data_);

    break;

  case MemoryType::DEVICE:
  case MemoryType::UNIFIED:
    cudaFree(data_);

    break;

  default:
    break;
  }

  data_ = nullptr;
  offset_ = 0;
  bytes_ = 0;
  memory_type_ = MemoryType::UNKNOWN_MEMORY;
  device_type_ = DeviceType::UNKNOWN_DEVICE;
}

void *Device::getData() { return data_; }
const void *Device::getData() const { return data_; }
uint64_t Device::getBytes() const { return bytes_; }
uint64_t Device::getOffset() const { return offset_; }
MemoryType Device::getMemoryType() const { return memory_type_; }
DeviceType Device::getDeviceType() const { return device_type_; }

void Device::setBytes(uint64_t bytes) {
  if (bytes_ == bytes)
    return;

  void *data = nullptr;
  MemoryType memory_type = (bytes_ > 0) ? MemoryType::PINNED : memory_type_;
  DeviceType device_type = (bytes_ > 0) ? DeviceType::CPU : device_type_;

  if (bytes > 0) {
    switch (memory_type) {
    case MemoryType::HOST:
      data = new (std::nothrow) uint8_t[bytes];

      break;

    case MemoryType::DEVICE:
      if (cudaMalloc(&data, bytes) != cudaSuccess)
        data = nullptr;

      break;

    case MemoryType::PINNED:
      if (cudaMallocHost(&data, bytes) != cudaSuccess)
        data = nullptr;

      break;

    case MemoryType::UNIFIED:
      if (cudaMallocManaged(&data, bytes) != cudaSuccess)
        data = nullptr;

      break;

    default:
      break;
    }

    if (data == nullptr)
      return;
  }

  if (bytes_ > 0 && data) {
    uint64_t copy_bytes = (bytes_ < bytes) ? bytes_ : bytes;
    DevicePair(this, this).copyData(data, data_, copy_bytes);
  }

  if (data_) {
    switch (memory_type_) {
    case MemoryType::HOST:
      delete[] static_cast<uint8_t *>(data_);

      break;

    case MemoryType::PINNED:
      cudaFreeHost(data_);

      break;

    case MemoryType::DEVICE:
    case MemoryType::UNIFIED:
      cudaFree(data_);

      break;

    default:
      break;
    }
  }

  data_ = data;
  bytes_ = bytes;
  memory_type_ = memory_type;
  device_type_ = device_type;

  if (offset_ > bytes_)
    offset_ = bytes_;
}

void *Device::makeData(uint64_t bytes) {
  if (offset_ + bytes > bytes_)
    return nullptr;

  void *data = static_cast<uint8_t *>(data_) + offset_;
  offset_ += bytes;

  return data;
}

void Device::freeData(uint64_t bytes) {
  if (bytes <= offset_)
    offset_ -= bytes;
  else
    offset_ = 0;
}
void Device::wipeData() { offset_ = 0; }

DevicePair::DevicePair(Device *first_device, Device *second_device) : first_device_(first_device), second_device_(second_device) {}

Device *DevicePair::getFirstDevice() { return first_device_; }
Device *DevicePair::getSecondDevice() { return second_device_; }
const Device *DevicePair::getFirstDevice() const { return first_device_; }
const Device *DevicePair::getSecondDevice() const { return second_device_; }

void DevicePair::copyData(void *destination, const void *source, uint64_t bytes) {
  if (!destination || !source || bytes == 0)
    return;

  auto destination_device_type = first_device_->getDeviceType();
  auto source_device_type = second_device_->getDeviceType();

  cudaMemcpyKind kind = cudaMemcpyDefault;

  if (source_device_type == DeviceType::CPU && destination_device_type == DeviceType::GPU)
    kind = cudaMemcpyHostToDevice;
  else if (source_device_type == DeviceType::GPU && destination_device_type == DeviceType::CPU)
    kind = cudaMemcpyDeviceToHost;
  else if (source_device_type == DeviceType::GPU && destination_device_type == DeviceType::GPU)
    kind = cudaMemcpyDeviceToDevice;
  else {
    std::memcpy(destination, source, bytes);
    return;
  }

  cudaMemcpy(destination, source, bytes, kind);
}

} // namespace startorch
