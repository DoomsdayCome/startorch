#pragma once

#include "startorch/common.hpp"

namespace startorch {
class Device {
private:
  void *data_ = nullptr;
  uint64_t bytes_ = 0;
  uint64_t offset_ = 0;
  MemoryType memory_type_ = MemoryType::UNKNOWN_MEMORY;
  DeviceType device_type_ = DeviceType::UNKNOWN_DEVICE;

public:
  Device() = default;
  Device(uint64_t size, MemoryType memory_type, DeviceType device_type);

  Device(const Device &other) = delete;
  Device &operator=(const Device &other) = delete;

  ~Device();

  Device(Device &&other) noexcept = delete;
  Device &operator=(Device &&other) noexcept = delete;

  void *getData();
  const void *getData() const;
  uint64_t getBytes() const;
  uint64_t getOffset() const;
  MemoryType getMemoryType() const;
  DeviceType getDeviceType() const;

  void setBytes(uint64_t bytes);

  void *makeData(uint64_t bytes);
  void freeData(uint64_t bytes);
  void wipeData();
};

inline Device AMD5625U = Device(1_GiB, MemoryType::HOST, DeviceType::CPU);
inline Device NVD3050M = Device(1_GiB, MemoryType::DEVICE, DeviceType::GPU);

class DevicePair {
private:
  Device *first_device_ = nullptr;
  Device *second_device_ = nullptr;

public:
  DevicePair() = default;
  DevicePair(Device *first_device, Device *second_device);

  DevicePair(const DevicePair &other) = default;
  DevicePair(DevicePair &&other) noexcept = default;

  ~DevicePair() = default;

  DevicePair &operator=(const DevicePair &other) = default;
  DevicePair &operator=(DevicePair &&other) noexcept = default;

  Device *getFirstDevice();
  Device *getSecondDevice();
  const Device *getFirstDevice() const;
  const Device *getSecondDevice() const;

  void copyData(void *destination, const void *source, uint64_t size);
};
} // namespace startorch
