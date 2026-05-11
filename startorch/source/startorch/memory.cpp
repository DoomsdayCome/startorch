#include "startorch/memory.hpp"
#include "startorch/common.hpp"
#include "startorch/device.hpp"
#include "startorch/format.hpp"

#include "darkside/assign.cuh"

#include <cstdint>
#include <cstring>
#include <new>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace startorch {

const void *Arena::getData() const { return data_; }
uint64_t Arena::getSize() const { return size_; }
uint64_t Arena::getOffset() const { return offset_; }
MemoryType Arena::getMemoryType() const { return memory_type_; }
const Device &Arena::getDevice() const { return device_; }

Arena::Arena(uint64_t size, MemoryType memory_type, const Device &device)
    : size_(size), memory_type_(memory_type), device_(device), data_(nullptr) {
  if (device_.getDeviceType() == DeviceType::CPU) {
    if (memory_type_ == MemoryType::DEVICE ||
        memory_type_ == MemoryType::UNIFIED)
      memory_type_ = MemoryType::HOST;
  } else if (device_.getDeviceType() == DeviceType::GPU) {
    if (memory_type_ == MemoryType::HOST ||
        memory_type_ == MemoryType::PINNED) {
      memory_type_ = MemoryType::DEVICE;
    }
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

void *Arena::makeData(uint64_t size) {
  if (offset_ + size > size_)
    return nullptr;

  void *pointer = (uint8_t *)data_ + offset_;

  offset_ += size;

  return pointer;
}

void Arena::freeData(uint64_t size) {
  if (size <= offset_)
    offset_ -= size;
  else
    offset_ = 0;
}

void Arena::wipeData() { offset_ = 0; }

void Arena::copyData(void *destination, const void *source, uint64_t size,
                     const DevicePair &device_pair) {
  if (!source || !destination || size == 0)
    return;

  cudaMemcpyKind kind = cudaMemcpyDefault;

  auto src_type = device_pair.getFirstDevice().getDeviceType();
  auto dst_type = device_pair.getSecondDevice().getDeviceType();

  if (src_type == DeviceType::CPU && dst_type == DeviceType::GPU)
    kind = cudaMemcpyHostToDevice;
  else if (src_type == DeviceType::GPU && dst_type == DeviceType::CPU)
    kind = cudaMemcpyDeviceToHost;
  else if (src_type == DeviceType::GPU && dst_type == DeviceType::GPU)
    kind = cudaMemcpyDeviceToDevice;
  else {
    memcpy(destination, source, size);
    return;
  }

  cudaMemcpy(destination, source, size, kind);
}

Storage::Storage(uint64_t size, ScalarType scalar_type, Arena *arena)
    : size_(size), scalar_type_(scalar_type), arena_(arena) {
  if (size_ == 0)
    return;

  if (arena == nullptr) {
    size_ = 0;
    return;
  }

  data_ = arena->makeData(size_ * darkside::getScalarTypeSize(scalar_type_));

  if (data_ == nullptr)
    size_ = 0;
}

Storage::Storage(const Storage &other)
    : size_(other.size_), scalar_type_(other.scalar_type_),
      arena_(other.arena_) {
  if (size_ == 0)
    return;

  data_ = arena_->makeData(size_ * darkside::getScalarTypeSize(scalar_type_));

  if (data_ == nullptr) {
    size_ = 0;
    return;
  }

  Arena::copyData(data_, other.data_,
                  size_ * darkside::getScalarTypeSize(scalar_type_),
                  DevicePair(arena_->getDevice(), arena_->getDevice()));
}

Storage::Storage(Storage &&other) noexcept
    : data_(other.data_), size_(other.size_), scalar_type_(other.scalar_type_),
      arena_(other.arena_) {
  data_ = nullptr;
  size_ = 0;
}

Storage::~Storage() {
  if (size_ == 0)
    return;

  if ((uint8_t *)data_ + size_ * darkside::getScalarTypeSize(scalar_type_) ==
      (uint8_t *)arena_->getData() + arena_->getOffset())
    arena_->freeData(size_ * darkside::getScalarTypeSize(scalar_type_));

  data_ = nullptr;
  size_ = 0;
}

Storage &Storage::operator=(const Storage &other) {
  if (this == &other)
    return *this;

  void *new_data = nullptr;
  uint64_t size = other.size_ * darkside::getScalarTypeSize(other.scalar_type_);

  if (other.arena_ && size > 0)
    new_data = other.arena_->makeData(size);

  if (size > 0 && new_data == nullptr)
    return *this;

  this->~Storage();

  arena_ = other.arena_;
  size_ = other.size_;
  scalar_type_ = other.scalar_type_;
  data_ = new_data;

  if (data_) {
    Arena::copyData(data_, other.data_, size,
                    DevicePair(other.arena_->getDevice(), arena_->getDevice()));
  }

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

const void *Storage::getData() const { return data_; }
uint64_t Storage::getSize() const { return size_; }
ScalarType Storage::getScalarType() const { return scalar_type_; };
Arena *Storage::getArena() const { return arena_; }

void Storage::setArena(Arena *arena) {
  if (arena == arena_ || size_ == 0) {
    arena_ = arena;
    return;
  }

  if (arena == nullptr)
    return;

  uint64_t size = size_ * darkside::getScalarTypeSize(scalar_type_);

  void *new_data = arena->makeData(size);
  if (new_data == nullptr)
    return;

  if (data_ != nullptr && arena_ != nullptr)
    Arena::copyData(new_data, data_, size,
                    DevicePair(arena_->getDevice(), arena->getDevice()));

  if ((uint8_t *)data_ + size ==
      (uint8_t *)arena_->getData() + arena_->getOffset())
    arena_->freeData(size);

  data_ = new_data;
  arena_ = arena;
}

void Storage::fillData(const darkside::ScalarValueToCPP &value) {
  switch (scalar_type_) {
#define FILL_DATA(T)                                                           \
  darkside::fillData<T>(data_, size_, value.value<T>(), arena_)
  case ScalarType::INT_8:
    FILL_DATA(int8_t);
    break;

  case ScalarType::INT_16:
    FILL_DATA(int16_t);
    break;

  case ScalarType::INT_32:
    FILL_DATA(int32_t);
    break;

  case ScalarType::INT_64:
    FILL_DATA(int64_t);
    break;

  case ScalarType::FLOAT_32:
    FILL_DATA(float);
    break;

  case ScalarType::FLOAT_64:
    FILL_DATA(double);
    break;

  case ScalarType::UNSIGNED_INT_8:
    FILL_DATA(uint8_t);
    break;

  case ScalarType::UNSIGNED_INT_16:
    FILL_DATA(uint16_t);
    break;

  case ScalarType::UNSIGNED_INT_32:
    FILL_DATA(uint32_t);
    break;

  case ScalarType::UNSIGNED_INT_64:
    FILL_DATA(uint64_t);
    break;

  default:
    break;

#undef FILL_DATA
  }
}

#define STORAGE_DISPATCH(TYPE_ENUM, FUNCTION, ...)                             \
  case ScalarType::INT_8:                                                      \
    FUNCTION<int8_t>(__VA_ARGS__);                                             \
    break;                                                                     \
  case ScalarType::INT_16:                                                     \
    FUNCTION<int16_t>(__VA_ARGS__);                                            \
    break;                                                                     \
  case ScalarType::INT_32:                                                     \
    FUNCTION<int32_t>(__VA_ARGS__);                                            \
    break;                                                                     \
  case ScalarType::INT_64:                                                     \
    FUNCTION<int64_t>(__VA_ARGS__);                                            \
    break;                                                                     \
  case ScalarType::FLOAT_32:                                                   \
    FUNCTION<float>(__VA_ARGS__);                                              \
    break;                                                                     \
  case ScalarType::FLOAT_64:                                                   \
    FUNCTION<double>(__VA_ARGS__);                                             \
    break;                                                                     \
  case ScalarType::UNSIGNED_INT_8:                                             \
    FUNCTION<uint8_t>(__VA_ARGS__);                                            \
    break;                                                                     \
  case ScalarType::UNSIGNED_INT_16:                                            \
    FUNCTION<uint16_t>(__VA_ARGS__);                                           \
    break;                                                                     \
  case ScalarType::UNSIGNED_INT_32:                                            \
    FUNCTION<uint32_t>(__VA_ARGS__);                                           \
    break;                                                                     \
  case ScalarType::UNSIGNED_INT_64:                                            \
    FUNCTION<uint64_t>(__VA_ARGS__);                                           \
    break;

void Storage::fillIncreaseData() {
  switch (scalar_type_) {
    STORAGE_DISPATCH(scalar_type_, darkside::fillIncreaseData, data_, size_,
                     arena_)
  default:
    break;
  }
}

void Storage::fillDecreaseData() {
  switch (scalar_type_) {
    STORAGE_DISPATCH(scalar_type_, darkside::fillDecreaseData, data_, size_,
                     arena_)
  default:
    break;
  }
}

#undef STORAGE_DISPATCH
} // namespace startorch
