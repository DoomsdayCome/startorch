#pragma once

#include "startorch/common.hpp"
#include "startorch/device.hpp"
#include "startorch/format.hpp"

#include <cstdint>

namespace startorch {
class Arena {
private:
  void *data_ = nullptr;
  uint64_t size_ = 0;
  uint64_t offset_ = 0;
  MemoryType memory_type_ = MemoryType::HOST;
  Device device_ = Device();

public:
  Arena() = default;
  Arena(uint64_t size, MemoryType memory_type, const Device &device);

  Arena(const Arena &other) = delete;
  Arena(Arena &&other) noexcept = delete;

  ~Arena();

  Arena &operator=(const Arena &other) = delete;
  Arena &operator=(Arena &&other) noexcept = delete;

  const void *getData() const;
  uint64_t getSize() const;
  uint64_t getOffset() const;
  MemoryType getMemoryType() const;
  const Device &getDevice() const;

  void *makeData(uint64_t size);
  void freeData(uint64_t size);
  void wipeData();

  static void copyData(void *destination, const void *source, uint64_t size,
                         const DevicePair &device_pair);
};

class Storage {
private:
  void *data_ = nullptr;
  uint64_t size_ = 0;
  ScalarType scalar_type_ = ScalarType::UNSIGNED_INT_8;
  Arena *arena_ = nullptr;

public:
  Storage() = default;
  Storage(uint64_t size, ScalarType scalar_type, Arena *arena);

  Storage(const Storage &other);
  Storage(Storage &&other) noexcept;

  ~Storage();

  Storage &operator=(const Storage &other);
  Storage &operator=(Storage &&other) noexcept;

  const void *getData() const;
  uint64_t getSize() const;
  ScalarType getScalarType() const;
  Arena *getArena() const;

  void setArena(Arena *arena);

  void fillData(const darkside::ScalarValueToCPP &value);
  void fillIncreaseData();
  void fillDecreaseData();
  void fillOrderData(OrderType order_type);
};
} // namespace startorch
