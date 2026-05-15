#include "startorch/layout.hpp"
#include "startorch/common.hpp"
#include "startorch/format.hpp"
#include "startorch/memory.hpp"

#include <cstdint>

namespace startorch {
Layout::Layout(const Storage &shape, const Storage &order, const Storage &strides, const Storage &offsets) {
  const ScalarType scalar_type = ScalarType::UNSIGNED_INT_64;
  const uint64_t size = shape.getSize();
  const uint64_t bytes = size * darkside::getScalarTypeSize(scalar_type);

  if (size == 0 || shape.getScalarType() != scalar_type || shape.getArena()->getDevice().getDeviceType() != DeviceType::CPU)
    return;

  Arena *arena = shape.getArena();
  shape_ = shape;
  bool broken = false;

  if (broken || order.getSize() != size || order.getScalarType() != scalar_type || order.getArena() != arena) {
    broken = true;
    order_ = Storage(bytes, scalar_type, arena);
    order_.fillIncreasedData(0, 1);
  } else
    order_ = order;

  if (broken || strides.getSize() != size || strides.getScalarType() != scalar_type || strides.getArena() != arena) {
    broken = true;
    strides_ = Storage(bytes, scalar_type, arena);

    uint64_t stride = 1;

    for (uint64_t i = size; i > 0; i--) {
      uint64_t dim = (order_.getData<uint64_t>())[i - 1];
      strides_.getData<uint64_t>()[dim] = stride;
      stride *= shape_.getData<uint64_t>()[dim];
    }

  } else
    strides_ = strides;

  if (broken || offsets.getSize() != size || offsets.getScalarType() != scalar_type || offsets.getArena() != arena) {
    broken = true;
    offsets_ = Storage(bytes, scalar_type, arena);
    offsets_.fillData(0);
  } else
    offsets_ = offsets;
}

Layout::Layout(const Storage &shape, OrderType order_type, const Storage &strides, const Storage &offsets) {
  const ScalarType scalar_type = ScalarType::UNSIGNED_INT_64;
  const uint64_t size = shape.getSize();
  const uint64_t bytes = size * darkside::getScalarTypeSize(scalar_type);

  if (size == 0 || shape.getScalarType() != scalar_type || shape.getArena()->getDevice().getDeviceType() != DeviceType::CPU)
    return;

  Arena *arena = shape.getArena();
  shape_ = shape;
  bool broken = false;

  order_ = Storage(bytes, scalar_type, arena);

  switch (order_type) {
  case OrderType::ROW_MAJOR:
    order_.fillIncreasedData(0, 1);
    break;

  case OrderType::COLUMN_MAJOR:
    order_.fillDecreasedData(size - 1, 1);
    break;

  default:
    break;
  }

  if (broken || strides.getSize() != size || strides.getScalarType() != scalar_type || strides.getArena() != arena) {
    broken = true;
    strides_ = Storage(bytes, scalar_type, arena);

    uint64_t stride = 1;

    for (uint64_t i = size; i > 0; i--) {
      uint64_t dim = (order_.getData<uint64_t>())[i - 1];
      strides_.getData<uint64_t>()[dim] = stride;
      stride *= shape_.getData<uint64_t>()[dim];
    }

  } else
    strides_ = strides;

  if (broken || offsets.getSize() != size || offsets.getScalarType() != scalar_type || offsets.getArena() != arena) {
    broken = true;
    offsets_ = Storage(bytes, scalar_type, arena);
    offsets_.fillData(0);
  } else
    offsets_ = offsets;
}

const Storage &Layout::getShape() const { return shape_; }
const Storage &Layout::getOrder() const { return order_; }
const Storage &Layout::getStrides() const { return strides_; }
const Storage &Layout::getOffsets() const { return offsets_; }

uint64_t Layout::getIndex(const Storage &indices) const {
  if (indices.getArena()->getDevice().getDeviceType() != DeviceType::CPU || indices.getScalarType() != ScalarType::UNSIGNED_INT_64 ||
      indices.getSize() != shape_.getSize())
    return 0;

  uint64_t index = 0;

  for (uint64_t i = 0; i < shape_.getSize(); i++)
    index += (indices.getData<uint64_t>()[i] + offsets_.getData<uint64_t>()[i]) * strides_.getData<uint64_t>()[i];

  return index;
}
} // namespace startorch
