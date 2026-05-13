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

void *Arena::getData() { return data_; }
uint64_t Arena::getSize() const { return size_; }
uint64_t Arena::getOffset() const { return offset_; }
MemoryType Arena::getMemoryType() const { return memory_type_; }
const Device &Arena::getDevice() const { return device_; }

Arena::Arena(uint64_t size, MemoryType memory_type, const Device &device)
    : size_(size), memory_type_(memory_type), device_(device), data_(nullptr) {

  if (device_.getDeviceType() == DeviceType::CPU) {
    if (memory_type_ == MemoryType::DEVICE ||
        memory_type_ == MemoryType::UNIFIED) {
      memory_type_ = MemoryType::HOST;
    }
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

void Arena::copyData(void *destination, const void *source, uint64_t size,
                     const DevicePair &device_pair) {

  if (!destination || !source || size == 0)
    return;

  auto src = device_pair.getFirstDevice().getDeviceType();
  auto dst = device_pair.getSecondDevice().getDeviceType();

  cudaMemcpyKind kind = cudaMemcpyDefault;

  if (src == DeviceType::CPU && dst == DeviceType::GPU) {
    kind = cudaMemcpyHostToDevice;
  } else if (src == DeviceType::GPU && dst == DeviceType::CPU) {
    kind = cudaMemcpyDeviceToHost;
  } else if (src == DeviceType::GPU && dst == DeviceType::GPU) {
    kind = cudaMemcpyDeviceToDevice;
  } else {
    memcpy(destination, source, size);
    return;
  }

  cudaMemcpy(destination, source, size, kind);
}

Storage::Storage(uint64_t size, ScalarType scalar_type, Arena *arena)
    : size_(size), scalar_type_(scalar_type), arena_(arena) {

  if (size_ == 0)
    return;

  if (arena_ == nullptr) {
    size_ = 0;
    return;
  }

  uint64_t byte = size_ * darkside::getScalarTypeSize(scalar_type_);

  data_ = arena_->makeData(byte);

  if (data_ == nullptr)
    size_ = 0;
}

Storage::Storage(const Storage &other)
    : size_(other.size_), scalar_type_(other.scalar_type_),
      arena_(other.arena_) {

  if (size_ == 0)
    return;

  uint64_t byte = size_ * darkside::getScalarTypeSize(scalar_type_);

  data_ = arena_->makeData(byte);

  if (data_ == nullptr) {
    size_ = 0;
    return;
  }

  Arena::copyData(data_, other.data_, byte,
                  DevicePair(arena_->getDevice(), arena_->getDevice()));
}

Storage::Storage(Storage &&other) noexcept
    : data_(other.data_), size_(other.size_), scalar_type_(other.scalar_type_),
      arena_(other.arena_) {

  other.data_ = nullptr;
  other.size_ = 0;
}

Storage::~Storage() {
  if (size_ == 0)
    return;

  uint64_t byte = size_ * darkside::getScalarTypeSize(scalar_type_);

  uint8_t *tail = (uint8_t *)arena_->getData() + arena_->getOffset();

  if ((uint8_t *)data_ + byte == tail)
    arena_->freeData(byte);

  data_ = nullptr;
  size_ = 0;
}

Storage &Storage::operator=(const Storage &other) {
  if (this == &other)
    return *this;

  uint64_t byte = other.size_ * darkside::getScalarTypeSize(other.scalar_type_);

  void *data = nullptr;

  if (other.arena_ && byte > 0)
    data = other.arena_->makeData(byte);

  if (byte > 0 && data == nullptr)
    return *this;

  this->~Storage();

  arena_ = other.arena_;
  size_ = other.size_;
  scalar_type_ = other.scalar_type_;
  data_ = data;

  if (data_) {
    Arena::copyData(data_, other.data_, byte,
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

void *Storage::getData() { return data_; }
uint64_t Storage::getSize() const { return size_; }
ScalarType Storage::getScalarType() const { return scalar_type_; }
Arena *Storage::getArena() const { return arena_; }

void Storage::setArena(Arena *arena) {
  if (arena == arena_ || size_ == 0) {
    arena_ = arena;
    return;
  }

  if (arena == nullptr)
    return;

  uint64_t byte = size_ * darkside::getScalarTypeSize(scalar_type_);

  void *data = arena->makeData(byte);

  if (data == nullptr)
    return;

  if (data_ && arena_) {
    Arena::copyData(data, data_, byte,
                    DevicePair(arena_->getDevice(), arena->getDevice()));
  }

  uint8_t *tail = (uint8_t *)arena_->getData() + arena_->getOffset();

  if ((uint8_t *)data_ + byte == tail)
    arena_->freeData(byte);

  data_ = data;
  arena_ = arena;
}

#define TARGET_DISPATCH(cpp_type, scalar_type)                                 \
  case scalar_type: {                                                          \
    using new_type =                                                           \
        typename darkside::ScalarTypeToCPPType<scalar_type>::getType;          \
                                                                               \
    darkside::convertDataType<cpp_type, new_type>(data_, new_data, size_,      \
                                                  arena_);                     \
    break;                                                                     \
  }

#define CONVERT_ALL(cpp_type)                                                  \
  TARGET_DISPATCH(cpp_type, ScalarType::INT_8)                                 \
  TARGET_DISPATCH(cpp_type, ScalarType::INT_16)                                \
  TARGET_DISPATCH(cpp_type, ScalarType::INT_32)                                \
  TARGET_DISPATCH(cpp_type, ScalarType::INT_64)                                \
  TARGET_DISPATCH(cpp_type, ScalarType::UNSIGNED_INT_8)                        \
  TARGET_DISPATCH(cpp_type, ScalarType::UNSIGNED_INT_16)                       \
  TARGET_DISPATCH(cpp_type, ScalarType::UNSIGNED_INT_32)                       \
  TARGET_DISPATCH(cpp_type, ScalarType::UNSIGNED_INT_64)                       \
  TARGET_DISPATCH(cpp_type, ScalarType::FLOAT_32)                              \
  TARGET_DISPATCH(cpp_type, ScalarType::FLOAT_64)

void Storage::setScalarType(ScalarType scalar_type) {
  if (scalar_type == scalar_type_ || size_ == 0) {
    scalar_type_ = scalar_type;
    return;
  }

  if (arena_ == nullptr)
    return;

  uint64_t byte = size_ * darkside::getScalarTypeSize(scalar_type);

  void *new_data = arena_->makeData(byte);

  if (new_data == nullptr)
    return;

#define CONVERT_CASE(src_type, cpp_type)                                       \
  case src_type: {                                                             \
    using T = cpp_type;                                                        \
                                                                               \
    switch (scalar_type) {                                                     \
      CONVERT_ALL(T)                                                           \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
                                                                               \
    break;                                                                     \
  }

  switch (scalar_type_) {
    CONVERT_CASE(ScalarType::INT_8, int8_t)
    CONVERT_CASE(ScalarType::INT_16, int16_t)
    CONVERT_CASE(ScalarType::INT_32, int32_t)
    CONVERT_CASE(ScalarType::INT_64, int64_t)

    CONVERT_CASE(ScalarType::UNSIGNED_INT_8, uint8_t)
    CONVERT_CASE(ScalarType::UNSIGNED_INT_16, uint16_t)
    CONVERT_CASE(ScalarType::UNSIGNED_INT_32, uint32_t)
    CONVERT_CASE(ScalarType::UNSIGNED_INT_64, uint64_t)

    CONVERT_CASE(ScalarType::FLOAT_32, float)
    CONVERT_CASE(ScalarType::FLOAT_64, double)

  default:
    break;
  }

#undef CONVERT_CASE
#undef CONVERT_ALL
#undef TARGET_DISPATCH

  uint64_t old_byte = size_ * darkside::getScalarTypeSize(scalar_type_);

  uint8_t *tail = (uint8_t *)arena_->getData() + arena_->getOffset();

  if ((uint8_t *)data_ + old_byte == tail)
    arena_->freeData(old_byte);

  data_ = new_data;
  scalar_type_ = scalar_type;
}

#define CASE_DISPATCH(scalar_type, cpp_type, action)                           \
  case scalar_type: {                                                          \
    using T = cpp_type;                                                        \
    action;                                                                    \
    break;                                                                     \
  }

#define STORAGE_DISPATCH(action)                                               \
  CASE_DISPATCH(ScalarType::INT_8, int8_t, action)                             \
  CASE_DISPATCH(ScalarType::INT_16, int16_t, action)                           \
  CASE_DISPATCH(ScalarType::INT_32, int32_t, action)                           \
  CASE_DISPATCH(ScalarType::INT_64, int64_t, action)                           \
  CASE_DISPATCH(ScalarType::UNSIGNED_INT_8, uint8_t, action)                   \
  CASE_DISPATCH(ScalarType::UNSIGNED_INT_16, uint16_t, action)                 \
  CASE_DISPATCH(ScalarType::UNSIGNED_INT_32, uint32_t, action)                 \
  CASE_DISPATCH(ScalarType::UNSIGNED_INT_64, uint64_t, action)                 \
  CASE_DISPATCH(ScalarType::FLOAT_32, float, action)                           \
  CASE_DISPATCH(ScalarType::FLOAT_64, double, action)

void Storage::fillData(const darkside::CPPValueToScalarValue &value) {

  switch (scalar_type_) {
    STORAGE_DISPATCH(
        darkside::fillData<T>(data_, size_, value.getValue<T>(), arena_))
  default:
    break;
  }
}

void Storage::fillIncreasedData(const darkside::CPPValueToScalarValue &start,
                                const darkside::CPPValueToScalarValue &step) {

  switch (scalar_type_) {
    STORAGE_DISPATCH(darkside::fillIncreasedData<T>(
        data_, size_, start.getValue<T>(), step.getValue<T>(), arena_))
  default:
    break;
  }
}

void Storage::fillDecreasedData(const darkside::CPPValueToScalarValue &start,
                                const darkside::CPPValueToScalarValue &step) {

  switch (scalar_type_) {
    STORAGE_DISPATCH(darkside::fillDecreasedData<T>(
        data_, size_, start.getValue<T>(), step.getValue<T>(), arena_))
  default:
    break;
  }
}

void Storage::fillOrderedData(const darkside::CPPValueToScalarValue &start,
                              const darkside::CPPValueToScalarValue &step,
                              OrderType order_type) {

  switch (scalar_type_) {
    STORAGE_DISPATCH(
        darkside::fillOrderedData<T>(data_, size_, start.getValue<T>(),
                                     step.getValue<int>(), order_type, arena_))
  default:
    break;
  }
}

#undef STORAGE_DISPATCH
#undef CASE_DISPATCH
} // namespace startorch
