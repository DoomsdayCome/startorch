#pragma once

#include "startorch/common.hpp"
#include "startorch/device.hpp"

#include <cstdint>

namespace startorch {
void *makeData(uint64_t size, const Device &device);
void freeData(void *pointer, const Device &device);
void copyData(void *destination, void *source, uint64_t size,
                const DevicePair &device_pair);

class Storage {
private:
  void *data_ = nullptr;
  uint64_t size_ = 0;
  ScalarType scalar_type_ = ScalarType::UNSIGNED_INT_8;
  Device device_ = Device();

public:
  Storage() = default;
  Storage(uint64_t size, ScalarType scalar_type,
          const Device &device);

  Storage(const Storage &other);
  Storage(Storage &&other) noexcept;

  ~Storage();

  Storage &operator=(const Storage &other);
  Storage &operator=(Storage &&other) noexcept;

  void *getData() const;
  uint64_t getSize() const;
  ScalarType getScalarType() const;
  const Device &getDevice() const;

  void setDevice(const Device &device);
};
} // namespace darkside
