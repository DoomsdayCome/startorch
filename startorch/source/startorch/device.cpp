#include "startorch/device.hpp"
#include "startorch/common.hpp"

namespace startorch {
Device::Device(DeviceType device_type) : device_type_(device_type) {}

bool Device::operator==(const Device &other) const {
  return device_type_ == other.device_type_;
}

bool Device::operator!=(const Device &other) const { return !(*this == other); }

DeviceType Device::getDeviceType() const { return device_type_; }

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
