#include "startorch/layout.hpp"
#include "startorch/common.hpp"
#include "startorch/format.hpp"
#include "startorch/memory.hpp"
#include <cstdint>
#include <cstring>

namespace startorch {
Layout::Layout(const Storage &shape, const Storage &order,
               const Storage &strides, const Storage &offsets,
               const Device &device) {
  if (shape.getSize() == 0 || shape.getScalarType() != scalar_type_)
    return;

  uint64_t size = shape.getSize();
  shape_ = shape;
  device_ = device; 

  auto is_valid = [&](const Storage &storage) {
    return (storage.getSize() == size &&
            storage.getScalarType() == scalar_type_);
  };

  Device cpu_dev = Device();

  if (is_valid(order))
    order_ = order;
  else {
    Device order_init_dev =
        (device_.getDeviceType() == DeviceType::CPU) ? device_ : cpu_dev;
    order_ = Storage(size, scalar_type_, order_init_dev);
    auto *order_pointer = (uint64_t *)order_.getData();
    for (uint64_t i = 0; i < size; i++)
      order_pointer[i] = i;
  }

  if (is_valid(strides))
    strides_ = strides;
  else {
    Device strides_init_dev =
        (device_.getDeviceType() == DeviceType::CPU) ? device_ : cpu_dev;
    strides_ = Storage(size, scalar_type_, strides_init_dev);
    auto *strides_pointer = (uint64_t *)strides_.getData();

    Storage temp_shape, temp_order;
    uint64_t *shape_pointer, *order_pointer;

    if (shape_.getDevice().getDeviceType() == DeviceType::CPU)
      shape_pointer = (uint64_t *)shape_.getData();
    else {
      temp_shape = Storage(size, scalar_type_, cpu_dev);
      copyData(temp_shape.getData(), shape_.getData(), size * sizeof(uint64_t),
               DevicePair(shape_.getDevice(), cpu_dev));
      shape_pointer = (uint64_t *)temp_shape.getData();
    }

    if (order_.getDevice().getDeviceType() == DeviceType::CPU)
      order_pointer = (uint64_t *)order_.getData();
    else {
      temp_order = Storage(size, scalar_type_, cpu_dev);
      copyData(temp_order.getData(), order_.getData(), size * sizeof(uint64_t),
               DevicePair(order_.getDevice(), cpu_dev));
      order_pointer = (uint64_t *)temp_order.getData();
    }

    uint64_t current_stride = 1;
    for (int64_t i = (int64_t)size - 1; i >= 0; i--) {
      uint64_t dim = order_pointer[i];
      strides_pointer[dim] = current_stride;
      current_stride *= shape_pointer[dim];
    }
  }

  if (is_valid(offsets)) {
    offsets_ = offsets;
  } else {
    Device offsets_init_dev =
        (device_.getDeviceType() == DeviceType::CPU) ? device_ : cpu_dev;
    offsets_ = Storage(size, scalar_type_, offsets_init_dev);
    memset(offsets_.getData(), 0, size * sizeof(uint64_t));
  }

  shape_.setDevice(device_);
  order_.setDevice(device_);
  strides_.setDevice(device_);
  offsets_.setDevice(device_);
}
} // namespace startorch
