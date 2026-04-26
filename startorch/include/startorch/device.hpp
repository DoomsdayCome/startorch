#pragma once

#include "startorch/common.hpp"

namespace startorch {
class Device {
private:
  DeviceType device_type_ = DeviceType::CPU;
  MemoryType memory_type_ = MemoryType::DEFAULT;

public:
  Device() = default;
  Device(DeviceType device_type, MemoryType memory_type);

  Device(const Device &other) = default;
  Device(Device &&other) noexcept = default;

  ~Device() = default;

  Device &operator=(const Device &other) = default;
  Device &operator=(Device &&other) noexcept = default;

  bool operator==(const Device &other) const;
  bool operator!=(const Device &other) const;

  DeviceType getDeviceType() const;
  MemoryType getMemoryType() const;
};

class DevicePair {
private:
  Device first_device_ = Device();
  Device second_device_ = Device();

public:
  DevicePair() = default;
  DevicePair(const Device &first_device, const Device &second_device);

  DevicePair(const DevicePair &other) = default;
  DevicePair(DevicePair &&other) noexcept = default;

  ~DevicePair() = default;

  DevicePair &operator=(const DevicePair &other) = default;
  DevicePair &operator=(DevicePair &&other) noexcept = default;

  bool operator==(const DevicePair &other) const;
  bool operator!=(const DevicePair &other) const;

  const Device &getFirstDevice() const;
  const Device &getSecondDevice() const;
};
} // namespace startorch
