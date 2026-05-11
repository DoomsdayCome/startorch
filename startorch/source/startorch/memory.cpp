#include "startorch/memory.hpp"
#include "startorch/common.hpp"
#include "startorch/device.hpp"
#include "startorch/format.hpp"

#include "darkside/kernel.cuh"

#include <cstdint>
#include <cstring>
#include <new>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

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

void *Storage::getData() { return data_; }
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

#define TARGET_DISPATCH(OLD_T, NEW_SCALAR_TYPE)                                \
  case ScalarType::NEW_SCALAR_TYPE: {                                          \
    using NEW_T = typename darkside::ScalarTypeToCPPType<                      \
        ScalarType::NEW_SCALAR_TYPE>::type;                                    \
    darkside::convertDataType<OLD_T, NEW_T>(data_, new_data, size_, arena_);    \
    break;                                                                     \
  }

#define CONVERT_ALL_TO_TARGET(OLD_T)                                           \
  TARGET_DISPATCH(OLD_T, INT_8)                                                \
  TARGET_DISPATCH(OLD_T, INT_16)                                               \
  TARGET_DISPATCH(OLD_T, INT_32)                                               \
  TARGET_DISPATCH(OLD_T, INT_64)                                               \
  TARGET_DISPATCH(OLD_T, UNSIGNED_INT_8)                                       \
  TARGET_DISPATCH(OLD_T, UNSIGNED_INT_16)                                      \
  TARGET_DISPATCH(OLD_T, UNSIGNED_INT_32)                                      \
  TARGET_DISPATCH(OLD_T, UNSIGNED_INT_64)                                      \
  TARGET_DISPATCH(OLD_T, FLOAT_32)                                             \
  TARGET_DISPATCH(OLD_T, FLOAT_64)

void Storage::setScalarType(ScalarType scalar_type) {
  if (scalar_type == scalar_type_ || size_ == 0) {
    scalar_type_ = scalar_type;
    return;
  }

  if (arena_ == nullptr)
    return;

  uint64_t new_elem_size = darkside::getScalarTypeSize(scalar_type);
  void *new_data = arena_->makeData(size_ * new_elem_size);

  if (new_data == nullptr)
    return;

  switch (scalar_type_) {
  case ScalarType::INT_8: {
    using T = int8_t;
    switch (scalar_type) { CONVERT_ALL_TO_TARGET(T) default : break; }
    break;
  }
  case ScalarType::INT_32: {
    using T = int32_t;
    switch (scalar_type) { CONVERT_ALL_TO_TARGET(T) default : break; }
    break;
  }
  case ScalarType::INT_64: {
    using T = int64_t;
    switch (scalar_type) { CONVERT_ALL_TO_TARGET(T) default : break; }
    break;
  }
  case ScalarType::FLOAT_32: {
    using T = float;
    switch (scalar_type) { CONVERT_ALL_TO_TARGET(T) default : break; }
    break;
  }
  case ScalarType::FLOAT_64: {
    using T = double;
    switch (scalar_type) { CONVERT_ALL_TO_TARGET(T) default : break; }
    break;
  }
  case ScalarType::UNSIGNED_INT_8: {
    using T = uint8_t;
    switch (scalar_type) { CONVERT_ALL_TO_TARGET(T) default : break; }
    break;
  }
  case ScalarType::UNSIGNED_INT_32: {
    using T = uint32_t;
    switch (scalar_type) { CONVERT_ALL_TO_TARGET(T) default : break; }
    break;
  }
  
  default:
    break;
  }

  uint64_t old_total_bytes = size_ * darkside::getScalarTypeSize(scalar_type_);
  if ((uint8_t *)data_ + old_total_bytes ==
      (uint8_t *)arena_->getData() + arena_->getOffset()) {
    arena_->freeData(old_total_bytes);
  }

  data_ = new_data;
  scalar_type_ = scalar_type;
}

#define STORAGE_DISPATCH(SCALAR_TYPE, ACTION)                                  \
  case ScalarType::INT_8: {                                                    \
    using T = int8_t;                                                          \
    ACTION;                                                                    \
    break;                                                                     \
  }                                                                            \
  case ScalarType::INT_16: {                                                   \
    using T = int16_t;                                                         \
    ACTION;                                                                    \
    break;                                                                     \
  }                                                                            \
  case ScalarType::INT_32: {                                                   \
    using T = int32_t;                                                         \
    ACTION;                                                                    \
    break;                                                                     \
  }                                                                            \
  case ScalarType::INT_64: {                                                   \
    using T = int64_t;                                                         \
    ACTION;                                                                    \
    break;                                                                     \
  }                                                                            \
  case ScalarType::FLOAT_32: {                                                 \
    using T = float;                                                           \
    ACTION;                                                                    \
    break;                                                                     \
  }                                                                            \
  case ScalarType::FLOAT_64: {                                                 \
    using T = double;                                                          \
    ACTION;                                                                    \
    break;                                                                     \
  }                                                                            \
  case ScalarType::UNSIGNED_INT_8: {                                           \
    using T = uint8_t;                                                         \
    ACTION;                                                                    \
    break;                                                                     \
  }                                                                            \
  case ScalarType::UNSIGNED_INT_16: {                                          \
    using T = uint16_t;                                                        \
    ACTION;                                                                    \
    break;                                                                     \
  }                                                                            \
  case ScalarType::UNSIGNED_INT_32: {                                          \
    using T = uint32_t;                                                        \
    ACTION;                                                                    \
    break;                                                                     \
  }                                                                            \
  case ScalarType::UNSIGNED_INT_64: {                                          \
    using T = uint64_t;                                                        \
    ACTION;                                                                    \
    break;                                                                     \
  }

void Storage::fillData(const darkside::CPPValueToScalarValue &value) {
  switch (scalar_type_) {
    STORAGE_DISPATCH(scalar_type_, darkside::fillData<T>(
                                       data_, size_, value.value<T>(), arena_))
  default:
    break;
  }
}

void Storage::fillIncreasedData(const darkside::CPPValueToScalarValue &start,
                                const darkside::CPPValueToScalarValue &step) {
  switch (scalar_type_) {
    STORAGE_DISPATCH(scalar_type_, darkside::fillIncreasedData<T>(
                                       data_, size_, start.value<T>(),
                                       step.value<T>(), arena_))
  default:
    break;
  }
}

void Storage::fillDecreasedData(const darkside::CPPValueToScalarValue &start,
                                const darkside::CPPValueToScalarValue &step) {
  switch (scalar_type_) {
    STORAGE_DISPATCH(scalar_type_, darkside::fillDecreasedData<T>(
                                       data_, size_, start.value<T>(),
                                       step.value<T>(), arena_))
  default:
    break;
  }
}

void Storage::fillOrderedData(const darkside::CPPValueToScalarValue &start,
                            const darkside::CPPValueToScalarValue &step,
                            OrderType order_type) {
  switch (scalar_type_) {
    STORAGE_DISPATCH(scalar_type_, darkside::fillOrderedData<T>(
                                       data_, size_, start.value<T>(),
                                       step.value<int>(), order_type, arena_))
  default:
    break;
  }
}
#undef STORAGE_DISPATCH
} // namespace startorch
