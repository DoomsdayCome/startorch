#pragma once

#include "startorch/common.hpp"
#include "startorch/device.hpp"
#include "startorch/format.hpp"

#include <cstdint>
#include <initializer_list>

namespace startorch {
class Arena {
private:
  void *data_ = nullptr;
  uint64_t size_ = 0;
  uint64_t offset_ = 0;
  MemoryType memory_type_ = MemoryType::UNKNOWN_MEMORY;
  Device device_ = Device();

public:
  Arena() = default;
  Arena(uint64_t size, MemoryType memory_type, const Device &device);

  Arena(const Arena &other) = delete;
  Arena(Arena &&other) noexcept = delete;

  ~Arena();

  Arena &operator=(const Arena &other) = delete;
  Arena &operator=(Arena &&other) noexcept = delete;

  void *getData();
  const void *getData() const;
  uint64_t getSize() const;
  uint64_t getOffset() const;
  MemoryType getMemoryType() const;
  const Device &getDevice() const;

  void setSize(uint64_t size);

  void *makeData(uint64_t size);
  void freeData(uint64_t size);
  void wipeData();

  static void copyData(void *destination, const void *source, uint64_t size, const DevicePair &device_pair);
};

inline Arena GLOBAL_CPU_ARENA = Arena(1_GiB, MemoryType::PINNED, Device(DeviceType::CPU));
inline Arena GLOBAL_GPU_ARENA = Arena(1_GiB, MemoryType::DEVICE, Device(DeviceType::GPU));

class Storage {
private:
  void *data_ = nullptr;
  uint64_t size_ = 0;
  ScalarType scalar_type_ = ScalarType::UNKNOWN_SCALAR;
  Arena *arena_ = nullptr;

public:
  Storage() = default;
  Storage(uint64_t size, ScalarType scalar_type, Arena *arena);
  Storage(std::initializer_list<Element> data, Arena *arena = &GLOBAL_CPU_ARENA);

  Storage(const Storage &other);
  Storage(Storage &&other) noexcept;

  ~Storage();

  Storage &operator=(const Storage &other);
  Storage &operator=(Storage &&other) noexcept;

  template <typename T> T *getData() { return (T *)data_; };
  template <typename T> const T *getData() const { return (T *)data_; };

  void *getData();
  const void *getData() const;
  uint64_t getSize() const;
  ScalarType getScalarType() const;
  Arena *getArena() const;

  void fillData(const Element &value);
  void fillIncreasedData(const Element &start, const Element &step);
  void fillDecreasedData(const Element &start, const Element &step);
};
} // namespace startorch
