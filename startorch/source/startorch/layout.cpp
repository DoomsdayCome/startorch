#include "startorch/layout.hpp"
#include "startorch/common.hpp"
#include "startorch/format.hpp"
#include "startorch/memory.hpp"

#include <algorithm>
#include <cstdint>

namespace startorch {
#define CONVERT_AND_COPY_STORAGE(dst, src)                                                                                                                     \
  if ((src).getArena() != arena && (src).getScalarType() == scalar_type) {                                                                                     \
    (dst) = Storage(size, (src).getScalarType(), arena);                                                                                                       \
    Arena::copyData((dst).getData(), (src).getData(), size * darkside::getScalarTypeSize((src).getScalarType()),                                               \
                    DevicePair((src).getArena()->getDevice(), arena->getDevice()));                                                                            \
  } else {                                                                                                                                                     \
    Storage temp_storage = Storage(size, scalar_type, arena);                                                                                                  \
                                                                                                                                                               \
    darkside::ScalarTypeToCPPType((src).getScalarType(), [&]<typename T>(darkside::CPPTypeToScalarType<T>) {                                                   \
      darkside::ScalarTypeToCPPType(scalar_type, [&]<typename N>(darkside::CPPTypeToScalarType<N>) {                                                           \
        for (uint64_t i = 0; i < size; i++)                                                                                                                    \
          temp_storage.getData<N>()[i] = static_cast<N>((src).getData<T>()[i]);                                                                                \
      });                                                                                                                                                      \
    });                                                                                                                                                        \
                                                                                                                                                               \
    (dst) = temp_storage;                                                                                                                                      \
  }

Layout::Layout(const Storage &shape, const Storage &order, const Storage &strides, const Storage &offsets) {
  uint64_t size = shape.getSize();

  if (size == 0)
    return;

  Arena *arena = nullptr;

  if (shape.getArena()->getDevice().getDeviceType() != DeviceType::CPU)
    arena = shape.getArena();
  else if (order.getSize() == size && order.getArena()->getDevice().getDeviceType() != DeviceType::CPU)
    arena = order.getArena();
  else if (strides.getSize() == size && strides.getArena()->getDevice().getDeviceType() != DeviceType::CPU)
    arena = strides.getArena();
  else if (offsets.getSize() == size && offsets.getArena()->getDevice().getDeviceType() != DeviceType::CPU)
    arena = offsets.getArena();
  else
    arena = &GLOBAL_CPU_ARENA;

  ScalarType scalar_type = ScalarType::UNKNOWN_SCALAR;

  if (shape.getScalarType() > scalar_type)
    scalar_type = shape.getScalarType();

  if (order.getSize() == size) {
    scalar_type = std::max(scalar_type, order.getScalarType());

    if (strides.getSize() == size) {
      scalar_type = std::max(scalar_type, strides.getScalarType());

      if (offsets.getSize() == size) {
        scalar_type = std::max(scalar_type, offsets.getScalarType());
      }
    }
  }

  if (scalar_type < ScalarType::INT_8)
    scalar_type = ScalarType::UNSIGNED_INT_64;

  uint64_t bytes = size * darkside::getScalarTypeSize(scalar_type);
  bool broken = false;

  if (shape.getArena() != arena || shape.getScalarType() != scalar_type) {
    CONVERT_AND_COPY_STORAGE(shape_, shape);
  } else
    shape_ = shape;

  if (order.getSize() != size) {
    broken = true;
    order_ = Storage(size, scalar_type, arena);
    order_.fillIncreasedData(0, 1);
  } else if (order.getArena() != arena || order.getScalarType() != scalar_type) {
    CONVERT_AND_COPY_STORAGE(order_, order);
  } else
    order_ = order;

  if (broken || strides.getSize() != size) {
    broken = true;
    strides_ = Storage(size, scalar_type, arena);

    darkside::ScalarTypeToCPPType(scalar_type, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
      uint64_t stride = 1;

      for (uint64_t i = size; i > 0; i--) {
        uint64_t dim = static_cast<uint64_t>(order_.getData<T>()[i - 1]);
        strides_.getData<T>()[dim] = static_cast<T>(stride);
        stride *= static_cast<uint64_t>(shape_.getData<T>()[dim]);
      }
    });
  } else if (strides.getArena() != arena || strides.getScalarType() != scalar_type) {
    CONVERT_AND_COPY_STORAGE(strides_, strides);
  } else
    strides_ = strides;

  if (broken || offsets.getSize() != size) {
    offsets_ = Storage(size, scalar_type, arena);
    offsets_.fillData(0);
  } else if (offsets.getArena() != arena || offsets.getScalarType() != scalar_type) {
    CONVERT_AND_COPY_STORAGE(offsets_, offsets);
  } else
    offsets_ = offsets;
}

Layout::Layout(const Storage &shape, OrderType order_type, const Storage &strides, const Storage &offsets) {
  uint64_t size = shape.getSize();

  if (size == 0)
    return;

  Arena *arena = nullptr;

  if (shape.getArena()->getDevice().getDeviceType() != DeviceType::CPU)
    arena = shape.getArena();
  else if (strides.getSize() == size && strides.getArena()->getDevice().getDeviceType() != DeviceType::CPU)
    arena = strides.getArena();
  else if (offsets.getSize() == size && offsets.getArena()->getDevice().getDeviceType() != DeviceType::CPU)
    arena = offsets.getArena();
  else
    arena = &GLOBAL_GPU_ARENA;

  ScalarType scalar_type = ScalarType::UNKNOWN_SCALAR;

  if (shape.getScalarType() > scalar_type)
    scalar_type = shape.getScalarType();

  if (strides.getSize() == size) {
    scalar_type = std::max(scalar_type, strides.getScalarType());

    if (offsets.getSize() == size)
      scalar_type = std::max(scalar_type, offsets.getScalarType());
  }

  if (scalar_type < ScalarType::INT_8)
    scalar_type = ScalarType::UNSIGNED_INT_64;

  uint64_t bytes = size * darkside::getScalarTypeSize(scalar_type);
  bool broken = false;

  if (shape.getArena() != arena || shape.getScalarType() != scalar_type) {
    CONVERT_AND_COPY_STORAGE(shape_, shape);
  } else
    shape_ = shape;

  order_ = Storage(size, scalar_type, arena);

  if (broken || strides.getSize() != size) {
    broken = true;
    strides_ = Storage(size, scalar_type, arena);

    darkside::ScalarTypeToCPPType(scalar_type, [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
      uint64_t stride = 1;

      for (uint64_t i = size; i > 0; i--) {
        uint64_t dim = static_cast<uint64_t>(order_.getData<T>()[i - 1]);
        strides_.getData<T>()[dim] = static_cast<T>(stride);
        stride *= static_cast<uint64_t>(shape_.getData<T>()[dim]);
      }
    });
  } else if (strides.getArena() != arena || strides.getScalarType() != scalar_type) {
    CONVERT_AND_COPY_STORAGE(strides_, strides);
  } else
    strides_ = strides;

  if (broken || offsets.getSize() != size) {
    offsets_ = Storage(size, scalar_type, arena);
    offsets_.fillData(0);
  } else if (offsets.getArena() != arena || offsets.getScalarType() != scalar_type) {
    CONVERT_AND_COPY_STORAGE(offsets_, offsets);
  } else
    offsets_ = offsets;
}

#undef CONVERT_AND_COPY_STORAGE

const Storage &Layout::getShape() const { return shape_; }
const Storage &Layout::getOrder() const { return order_; }
const Storage &Layout::getStrides() const { return strides_; }
const Storage &Layout::getOffsets() const { return offsets_; }

uint64_t Layout::getIndex(const Storage &indices) const {
  if (shape_.getSize() == 0)
    return 0;

  if (indices.getSize() != shape_.getSize())
    return 0;

  const Storage *indices_ = nullptr;
  Storage temp = Storage();

  if (indices.getArena()->getDevice().getDeviceType() != DeviceType::CPU) {
    temp = Storage(indices.getSize(), indices.getScalarType(), strides_.getArena());
    Arena::copyData(temp.getData(), indices.getData(), indices.getSize() * darkside::getScalarTypeSize(indices.getScalarType()),
                    DevicePair(indices.getArena()->getDevice(), strides_.getArena()->getDevice()));
    indices_ = &temp;
  } else
    indices_ = &indices;

  uint64_t index = 0;

  darkside::ScalarTypeToCPPType(indices_->getScalarType(), [&]<typename T>(darkside::CPPTypeToScalarType<T>) {
    darkside::ScalarTypeToCPPType(strides_.getScalarType(), [&]<typename N>(darkside::CPPTypeToScalarType<N>) {
      for (uint64_t i = 0; i < shape_.getSize(); i++)
        index += (static_cast<uint64_t>(indices_->getData<T>()[i]) + static_cast<uint64_t>(offsets_.getData<N>()[i])) *
                 static_cast<uint64_t>(strides_.getData<N>()[i]);
    });
  });

  return index;
}
} // namespace startorch
