#include "startorch/device.hpp"
#include "startorch/common.hpp"

namespace startorch {
Device::Device(DeviceType device_type, MemoryType memory_type)
    : device_type_(device_type), memory_type_(MemoryType::DEFAULT) {
  switch (memory_type) {
  case MemoryType::PINNED:
    if (device_type_ == DeviceType::CPU)
      memory_type_ = memory_type;
    break;

  case MemoryType::UNIFIED:
    if (device_type_ == DeviceType::GPU)
      memory_type_ = memory_type;
    break;

  default:
    break;
  }
}

bool Device::operator==(const Device &other) const {
  return device_type_ == other.device_type_ &&
         memory_type_ == other.memory_type_;
}

bool Device::operator!=(const Device &other) const { return !(*this == other); }

DeviceType Device::getDeviceType() const { return device_type_; }
MemoryType Device::getMemoryType() const { return memory_type_; }

DevicePair::DevicePair(const Device &first_device, const Device &second_device)
    : first_device_(first_device), second_device_(second_device) {}

bool DevicePair::operator==(const DevicePair &other) const {
  return first_device_ == other.first_device_ &&
         second_device_ == other.second_device_;
}
bool DevicePair::operator!=(const DevicePair &other) const {
  return !(*this == other);
}

const Device &DevicePair::getFirstDevice() const { return first_device_; }
const Device &DevicePair::getSecondDevice() const { return second_device_; }

} // namespace startorch
