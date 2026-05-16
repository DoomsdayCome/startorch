#include "startorch/memory.hpp"
#include "startorch/common.hpp"
#include "startorch/device.hpp"
#include "startorch/format.hpp"

#include "darkside/kernel.cuh"

#include <cstdint>
#include <cstring>
#include <new>

#include <cuda_runtime.h>

namespace startorch {
Arena::Arena(uint64_t size, MemoryType memory_type, const Device &device) : size_(size), memory_type_(memory_type), device_(device), data_(nullptr) {
  if (device_.getDeviceType() == DeviceType::CPU) {
    if (memory_type_ == MemoryType::DEVICE || memory_type_ == MemoryType::UNIFIED)
      memory_type_ = MemoryType::HOST;
  } else if (device_.getDeviceType() == DeviceType::GPU) {
    if (memory_type_ == MemoryType::HOST || memory_type_ == MemoryType::PINNED)
      memory_type_ = MemoryType::DEVICE;
  }

  if (size_ == 0)
    return;

  switch (memory_type_) {
  case MemoryType::HOST:
    data_ = new (std::nothrow) uint8_t[size_];
    break;

  case MemoryType::DEVICE:
    if (cudaMalloc(&data_, size_) != cudaSuccess)
      data_ = nullptr;
    break;

  case MemoryType::PINNED:
    if (cudaMallocHost(&data_, size_) != cudaSuccess)
      data_ = nullptr;
    break;

  case MemoryType::UNIFIED:
    if (cudaMallocManaged(&data_, size_) != cudaSuccess)
      data_ = nullptr;
    break;

  default:
    break;
  }

  if (data_ == nullptr)
    size_ = 0;
}

Arena::~Arena() {
  if (size_ == 0)
    return;

  switch (memory_type_) {
  case MemoryType::HOST:
    delete[] (uint8_t *)data_;
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
  size_ = 0;
}

void *Arena::getData() { return data_; }
const void *Arena::getData() const { return data_; }
uint64_t Arena::getSize() const { return size_; }
uint64_t Arena::getOffset() const { return offset_; }
MemoryType Arena::getMemoryType() const { return memory_type_; }
const Device &Arena::getDevice() const { return device_; }

void Arena::setSize(uint64_t size) {
  if (size == size_)
    return;

  void *new_data = nullptr;
  if (size > 0) {
    switch (memory_type_) {
    case MemoryType::HOST:
      new_data = new (std::nothrow) uint8_t[size];
      break;

    case MemoryType::DEVICE:
      if (cudaMalloc(&new_data, size) != cudaSuccess)
        new_data = nullptr;
      break;

    case MemoryType::PINNED:
      if (cudaMallocHost(&new_data, size) != cudaSuccess)
        new_data = nullptr;
      break;

    case MemoryType::UNIFIED:
      if (cudaMallocManaged(&new_data, size) != cudaSuccess)
        new_data = nullptr;
      break;

    default:
      break;
    }
  }

  if (data_ && new_data) {
    uint64_t copy_size = (size < size_) ? size : size_;
    copyData(new_data, data_, copy_size, DevicePair(device_, device_));
  }

  if (data_) {
    switch (memory_type_) {
    case MemoryType::HOST:
      delete[] (uint8_t *)data_;
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

  data_ = new_data;
  size_ = (data_ == nullptr && size > 0) ? 0 : size;

  if (offset_ > size_)
    offset_ = size_;
}

void *Arena::makeData(uint64_t size) {
  if (offset_ + size > size_)
    return nullptr;

  void *data = (uint8_t *)data_ + offset_;

  offset_ += size;

  return data;
}

void Arena::freeData(uint64_t size) {
  if (size <= offset_)
    offset_ -= size;
  else
    offset_ = 0;
}

void Arena::wipeData() { offset_ = 0; }

void Arena::copyData(void *destination, const void *source, uint64_t size, const DevicePair &device_pair) {
  if (!destination || !source || size == 0)
    return;

  auto src = device_pair.getFirstDevice().getDeviceType();
  auto dst = device_pair.getSecondDevice().getDeviceType();

  cudaMemcpyKind kind = cudaMemcpyDefault;

  if (src == DeviceType::CPU && dst == DeviceType::GPU)
    kind = cudaMemcpyHostToDevice;
  else if (src == DeviceType::GPU && dst == DeviceType::CPU)
    kind = cudaMemcpyDeviceToHost;
  else if (src == DeviceType::GPU && dst == DeviceType::GPU)
    kind = cudaMemcpyDeviceToDevice;
  else {
    memcpy(destination, source, size);
    return;
  }

  cudaMemcpy(destination, source, size, kind);
}

Storage::Storage(uint64_t size, ScalarType scalar_type, Arena *arena) : size_(size), scalar_type_(scalar_type), arena_(arena) {
  if (size_ == 0)
    return;

  if (arena_ == nullptr) {
    size_ = 0;
    return;
  }

  uint64_t bytes = size_ * darkside::getScalarTypeSize(scalar_type_);

  data_ = arena_->makeData(bytes);

  if (data_ == nullptr)
    size_ = 0;
}

Storage::Storage(const Storage &other) : size_(other.size_), scalar_type_(other.scalar_type_), arena_(other.arena_) {
  if (size_ == 0)
    return;

  uint64_t bytes = size_ * darkside::getScalarTypeSize(scalar_type_);

  data_ = arena_->makeData(bytes);

  if (data_ == nullptr) {
    size_ = 0;
    return;
  }

  Arena::copyData(data_, other.data_, bytes, DevicePair(arena_->getDevice(), arena_->getDevice()));
}

Storage::Storage(Storage &&other) noexcept : data_(other.data_), size_(other.size_), scalar_type_(other.scalar_type_), arena_(other.arena_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

Storage::~Storage() {
  if (size_ == 0)
    return;

  uint64_t bytes = size_ * darkside::getScalarTypeSize(scalar_type_);

  uint8_t *tail = (uint8_t *)arena_->getData() + arena_->getOffset();

  if ((uint8_t *)data_ + bytes == tail)
    arena_->freeData(bytes);

  data_ = nullptr;
  size_ = 0;
}

Storage &Storage::operator=(const Storage &other) {
  if (this == &other)
    return *this;

  uint64_t bytes = other.size_ * darkside::getScalarTypeSize(other.scalar_type_);

  void *data = nullptr;

  if (other.arena_ && bytes > 0)
    data = other.arena_->makeData(bytes);

  if (bytes > 0 && data == nullptr)
    return *this;

  this->~Storage();

  arena_ = other.arena_;
  size_ = other.size_;
  scalar_type_ = other.scalar_type_;
  data_ = data;

  if (data_)
    Arena::copyData(data_, other.data_, bytes, DevicePair(other.arena_->getDevice(), arena_->getDevice()));

  return *this;
}

Storage &Storage::operator=(Storage &&other) noexcept {
  if (this == &other)
    return *this;

  this->~Storage();

  data_ = other.data_;
  size_ = other.size_;
  scalar_type_ = other.scalar_type_;
  arena_ = other.arena_;

  other.data_ = nullptr;
  other.size_ = 0;

  return *this;
}

void *Storage::getData() { return data_; }
const void *Storage::getData() const { return data_; }
uint64_t Storage::getSize() const { return size_; }
ScalarType Storage::getScalarType() const { return scalar_type_; }
Arena *Storage::getArena() const { return arena_; }

void Storage::fillData(const Element &value) {
  if (size_ == 0)
    return;

  darkside::ScalarTypeToCPPType(scalar_type_,
                                [&]<typename T>(darkside::CPPTypeToScalarType<T>) { darkside::fillData<T>((T *)data_, size_, *value.getData<T>(), arena_); });
}

void Storage::fillIncreasedData(const Element &start, const Element &step) {
  if (size_ == 0)
    return;

  darkside::ScalarTypeToCPPType(scalar_type_, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
    darkside::fillIncreasedData<T>((T *)data_, size_, *start.getData<T>(), *step.getData<T>(), arena_);
  });
}

void Storage::fillDecreasedData(const Element &start, const Element &step) {
  if (size_ == 0)
    return;

  darkside::ScalarTypeToCPPType(scalar_type_, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
    darkside::fillDecreasedData<T>((T *)data_, size_, *start.getData<T>(), *step.getData<T>(), arena_);
  });
}
} // namespace startorch
