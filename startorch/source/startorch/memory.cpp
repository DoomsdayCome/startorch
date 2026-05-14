#include "startorch/memory.hpp"
#include "startorch/common.hpp"
#include "startorch/device.hpp"
#include "startorch/format.hpp"
#include "startorch/logger.hpp"

#include "darkside/kernel.cuh"

#include <cstdint>
#include <cstring>
#include <new>

#include <cuda_runtime.h>

namespace startorch {
Arena::Arena(uint64_t size, MemoryType memory_type, const Device &device) : size_(size), memory_type_(memory_type), device_(device), data_(nullptr) {
  if (device_.getDeviceType() == DeviceType::CPU) {
    if (memory_type_ == MemoryType::DEVICE || memory_type_ == MemoryType::UNIFIED) {
      Logger::makeLog(
          LoggerType::WARN, __FILE__, __LINE__,
          "Arena constructor: The memory type is incompatible with device type [CPU: AMD Ryzen 5 5625U (12) @ 4.39 GHz]. Forced memmory type to [HOST].");
      memory_type_ = MemoryType::HOST;
    }
  } else if (device_.getDeviceType() == DeviceType::GPU) {
    if (memory_type_ == MemoryType::HOST || memory_type_ == MemoryType::PINNED) {
      Logger::makeLog(LoggerType::WARN, __FILE__, __LINE__,
                      "Arena constructor: The memory type is incompatible with device type [GPU: NVIDIA GeForce RTX 3050 Mobile [Discrete]]. Forced memmory "
                      "type to [DEVICE].");
      memory_type_ = MemoryType::DEVICE;
    }
  }

  if (size_ == 0) {
    Logger::makeLog(LoggerType::WARN, __FILE__, __LINE__,
                    "Arena constructor: The size [0 bytes] could not be initialized. Returning immediately and keeping the default state.");
    return;
  }

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

  if (data_ == nullptr) {
    Logger::makeLog(LoggerType::WARN, __FILE__, __LINE__,
                    "Arena constructor: The data [nullptr] failed to initialize. Returning immediately and keeping the default state.");
    size_ = 0;
  }

  Logger::makeLog(LoggerType::INFO, __FILE__, __LINE__, "Arena constructor: The arena constructed successfully.");
}

Arena::~Arena() {
  if (size_ == 0) {
    Logger::makeLog(LoggerType::WARN, __FILE__, __LINE__, "Arena destructor: The size is [0 bytes]. Skipping destruction.");
    return;
  }

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

  Logger::makeLog(LoggerType::INFO, __FILE__, __LINE__, "Arena destructor: The arena destructed successfully.");
}

void *Arena::getData() { return data_; }
const void *Arena::getData() const { return data_; }
uint64_t Arena::getSize() const { return size_; }
uint64_t Arena::getOffset() const { return offset_; }
MemoryType Arena::getMemoryType() const { return memory_type_; }
const Device &Arena::getDevice() const { return device_; }

void *Arena::makeData(uint64_t size) {
  if (offset_ + size > size_) {
    Logger::makeLog(LoggerType::WARN, __FILE__, __LINE__, "Arena make data: Requested allocation size exceeds remaining arena capacity.");
    return nullptr;
  }

  void *data = (uint8_t *)data_ + offset_;

  offset_ += size;

  Logger::makeLog(LoggerType::INFO, __FILE__, __LINE__, "Arena make data: Memory allocated successfully.");

  return data;
}

void Arena::freeData(uint64_t size) {
  if (size <= offset_) {
    Logger::makeLog(LoggerType::INFO, __FILE__, __LINE__, "Arena free data: Freed memory successfully.");
    offset_ -= size;
  } else {
    Logger::makeLog(LoggerType::WARN, __FILE__, __LINE__, "Arena free data: The requested free size exceeds current offset. Resetting offset to [0].");
    offset_ = 0;
  }
}

void Arena::wipeData() {
  offset_ = 0;
  Logger::makeLog(LoggerType::INFO, __FILE__, __LINE__, "Arena wipe data: The offset reset successfully.");
}

void Arena::copyData(void *destination, const void *source, uint64_t size, const DevicePair &device_pair) {

  if (!destination || !source || size == 0) {
    Logger::makeLog(LoggerType::WARN, __FILE__, __LINE__, "Arena copy data: Invalid copy arguments detected.");
    return;
  }

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

  Logger::makeLog(LoggerType::INFO, __FILE__, __LINE__, "Arena copy data: Data copied successfully.");
}

Storage::Storage(uint64_t size, ScalarType scalar_type, Arena *arena) : size_(size), scalar_type_(scalar_type), arena_(arena) {
  if (size_ == 0) {
    Logger::makeLog(LoggerType::WARN, __FILE__, __LINE__,
                    "Storage constructor: The size [0 bytes] could not be initialized. Returning immediately and keeping the default state.");
    return;
  }

  if (arena_ == nullptr) {
    Logger::makeLog(LoggerType::WARN, __FILE__, __LINE__,
                    "Storage constructor: The arena [nullptr] could not be initialized. Returning immediately and keeping the default state.");
    size_ = 0;
    return;
  }

  uint64_t byte = size_ * darkside::getScalarTypeSize(scalar_type_);

  data_ = arena_->makeData(byte);

  if (data_ == nullptr) {
    Logger::makeLog(LoggerType::WARN, __FILE__, __LINE__,
                    "Storage constructor: The data [nullptr] failed to initialize. Returning immediately and keeping the default state.");
    size_ = 0;
  }

  Logger::makeLog(LoggerType::INFO, __FILE__, __LINE__, "Storage constructor: Storage constructed successfully.");
}

Storage::Storage(const Storage &other) : size_(other.size_), scalar_type_(other.scalar_type_), arena_(other.arena_) {
  if (size_ == 0) {
    Logger::makeLog(LoggerType::WARN, __FILE__, __LINE__,
                    "Storage copy constructor: The source size [0] failed to copy. Returning immediately and keeping the default state.");
    return;
  }

  uint64_t byte = size_ * darkside::getScalarTypeSize(scalar_type_);

  data_ = arena_->makeData(byte);

  if (data_ == nullptr) {
    Logger::makeLog(LoggerType::WARN, __FILE__, __LINE__,
                    "Storage copy constructor: The data [nullptr] failed to initialize. Returning immediately and keeping the default state.");
    size_ = 0;
    return;
  }

  Arena::copyData(data_, other.data_, byte, DevicePair(arena_->getDevice(), arena_->getDevice()));

  Logger::makeLog(LoggerType::INFO, __FILE__, __LINE__, "Storage copy constructor: Storage coppied successfully.");
}

Storage::Storage(Storage &&other) noexcept : data_(other.data_), size_(other.size_), scalar_type_(other.scalar_type_), arena_(other.arena_) {

  other.data_ = nullptr;
  other.size_ = 0;

  Logger::makeLog(LoggerType::INFO, __FILE__, __LINE__, "Storage move constructor: Storage moved successfully.");
}

Storage::~Storage() {
  if (size_ == 0) {
    Logger::makeLog(LoggerType::WARN, __FILE__, __LINE__, "Storage destructor destructor: The size is [0 bytes]. Skipping destruction.");
    return;
  }

  uint64_t byte = size_ * darkside::getScalarTypeSize(scalar_type_);

  uint8_t *tail = (uint8_t *)arena_->getData() + arena_->getOffset();

  if ((uint8_t *)data_ + byte == tail)
    arena_->freeData(byte);

  data_ = nullptr;
  size_ = 0;

  Logger::makeLog(LoggerType::INFO, __FILE__, __LINE__, "Arena destructor: The arena destructed successfully.");
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
    Arena::copyData(data_, other.data_, byte, DevicePair(other.arena_->getDevice(), arena_->getDevice()));
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

#define CASE_DISPATCH(scalar_type, cpp_type, action)                                                                                                           \
  case scalar_type: {                                                                                                                                          \
    using T = cpp_type;                                                                                                                                        \
    action;                                                                                                                                                    \
    break;                                                                                                                                                     \
  }

#define STORAGE_DISPATCH(action)                                                                                                                               \
  CASE_DISPATCH(ScalarType::INT_8, int8_t, action)                                                                                                             \
  CASE_DISPATCH(ScalarType::INT_16, int16_t, action)                                                                                                           \
  CASE_DISPATCH(ScalarType::INT_32, int32_t, action)                                                                                                           \
  CASE_DISPATCH(ScalarType::INT_64, int64_t, action)                                                                                                           \
  CASE_DISPATCH(ScalarType::UNSIGNED_INT_8, uint8_t, action)                                                                                                   \
  CASE_DISPATCH(ScalarType::UNSIGNED_INT_16, uint16_t, action)                                                                                                 \
  CASE_DISPATCH(ScalarType::UNSIGNED_INT_32, uint32_t, action)                                                                                                 \
  CASE_DISPATCH(ScalarType::UNSIGNED_INT_64, uint64_t, action)                                                                                                 \
  CASE_DISPATCH(ScalarType::FLOAT_32, float, action)                                                                                                           \
  CASE_DISPATCH(ScalarType::FLOAT_64, double, action)

void Storage::fillData(const Element &value) {

  switch (scalar_type_) {
    STORAGE_DISPATCH(darkside::fillData<T>(data_, size_, value.getData<T>(), arena_))
  default:
    break;
  }
}

void Storage::fillIncreasedData(const Element &start, const Element &step) {

  switch (scalar_type_) {
    STORAGE_DISPATCH(darkside::fillIncreasedData<T>(data_, size_, start.getData<T>(), step.getData<T>(), arena_))
  default:
    break;
  }
}

void Storage::fillDecreasedData(const Element &start, const Element &step) {

  switch (scalar_type_) {
    STORAGE_DISPATCH(darkside::fillDecreasedData<T>(data_, size_, start.getData<T>(), step.getData<T>(), arena_))
  default:
    break;
  }
}

#undef STORAGE_DISPATCH
#undef CASE_DISPATCH
} // namespace startorch
