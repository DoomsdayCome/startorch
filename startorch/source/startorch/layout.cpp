#include "startorch/layout.hpp"
#include "startorch/common.hpp"
#include "startorch/format.hpp"
#include "startorch/memory.hpp"

#include "darkside/kernel.cuh"

#include <cstdint>

namespace startorch {
Layout::Layout(const Storage &shape, const Storage &order,
               const Storage &strides, const Storage &offsets, Arena *arena)
    : arena_(arena), size_(shape.getSize()) {

  if (arena_ == nullptr)
    return;

  if (size_ == 0) {
    shape_.setArena(arena_);
    order_.setArena(arena_);
    strides_.setArena(arena_);
    offsets_.setArena(arena_);

    return;
  } else {
    shape_ = shape;
    shape_.setScalarType(scalar_type_);
    shape_.setArena(arena_);
  }

  if (order.getSize() != size_) {
    order_ = Storage(size_, scalar_type_, arena_);
    order_.fillOrderedData(0, 1, order_type_);
  } else {
    order_ = order;
    order_.setScalarType(scalar_type_);

    void *read_pointer = nullptr;
    uint64_t bytes = size_ * darkside::getScalarTypeSize(scalar_type_);
    Arena temp(bytes, MemoryType::HOST, DeviceType::CPU);

    if (order_.getArena()->getDevice().getDeviceType() == DeviceType::CPU)
      read_pointer = order_.getData();
    else if (arena_->getDevice().getDeviceType() == DeviceType::CPU) {
      order_.setArena(arena_);
      read_pointer = order_.getData();
    } else {
      order_.setArena(&temp);
      read_pointer = order_.getData();
    }

    switch (scalar_type_) {
#define ORDER_TYPE_CASE(scalar_type, cpp_type)                                 \
  case ScalarType::scalar_type: {                                              \
    cpp_type *check_pointer = (cpp_type *)read_pointer;                        \
    bool increasing = true;                                                    \
    bool decreasing = true;                                                    \
                                                                               \
    for (uint64_t i = 0; i < size_ - 1; i++) {                                 \
      if (check_pointer[i] > check_pointer[i + 1])                             \
        increasing = false;                                                    \
      if (check_pointer[i] < check_pointer[i + 1])                             \
        decreasing = false;                                                    \
    }                                                                          \
                                                                               \
    if (increasing)                                                            \
      order_type_ = OrderType::ROW_MAJOR;                                      \
    else if (decreasing)                                                       \
      order_type_ = OrderType::COLUMN_MAJOR;                                   \
    else                                                                       \
      order_type_ = OrderType::NONE_MAJOR;                                     \
    break;                                                                     \
  }

      ORDER_TYPE_CASE(INT_8, int8_t)
      ORDER_TYPE_CASE(INT_16, int16_t)
      ORDER_TYPE_CASE(INT_32, int32_t)
      ORDER_TYPE_CASE(INT_64, int64_t)
      ORDER_TYPE_CASE(UNSIGNED_INT_8, uint8_t)
      ORDER_TYPE_CASE(UNSIGNED_INT_16, uint16_t)
      ORDER_TYPE_CASE(UNSIGNED_INT_32, uint32_t)
      ORDER_TYPE_CASE(UNSIGNED_INT_64, uint64_t)
      ORDER_TYPE_CASE(FLOAT_32, float)
      ORDER_TYPE_CASE(FLOAT_64, double)

#undef ORDER_TYPE_CASE

    default:
      break;
    }

    order_.setArena(arena_);
  }

  if (strides.getSize() != size_) {
    strides_ = Storage(size_, scalar_type_, arena_);

    switch (scalar_type_) {
#define STRIDE_CASE(ST_TYPE, CPP_TYPE)                                         \
  case ScalarType::ST_TYPE:                                                    \
    darkside::fillStrides<CPP_TYPE>(strides_.getData(), shape_.getData(),      \
                                    order_.getData(), size_, arena_);          \
    break;

      STRIDE_CASE(INT_8, int8_t)
      STRIDE_CASE(INT_16, int16_t)
      STRIDE_CASE(INT_32, int32_t)
      STRIDE_CASE(INT_64, int64_t)
      STRIDE_CASE(UNSIGNED_INT_8, uint8_t)
      STRIDE_CASE(UNSIGNED_INT_16, uint16_t)
      STRIDE_CASE(UNSIGNED_INT_32, uint32_t)
      STRIDE_CASE(UNSIGNED_INT_64, uint64_t)
      STRIDE_CASE(FLOAT_32, float)
      STRIDE_CASE(FLOAT_64, double)

#undef STRIDE_CASE

    default:
      break;
    }
  } else {
    strides_ = strides;
    strides_.setScalarType(scalar_type_);
    strides_.setArena(arena_);
  }

  if (offsets.getSize() != size_) {
    offsets_ = Storage(size_, scalar_type_, arena_);
    offsets_.fillData(0);
  } else {
    offsets_ = offsets;
    offsets_.setScalarType(scalar_type_);
    offsets_.setArena(arena_);
  }
}

Layout::Layout(const Storage &shape, OrderType order_type,
               const Storage &strides, const Storage &offsets, Arena *arena)
    : arena_(arena), size_(shape.getSize()), order_type_(order_type) {

  if (arena_ == nullptr)
    return;

  if (size_ == 0) {
    shape_.setArena(arena_);
    order_.setArena(arena_);
    strides_.setArena(arena_);
    offsets_.setArena(arena_);

    return;
  } else {
    shape_ = shape;
    shape_.setScalarType(scalar_type_);
    shape_.setArena(arena_);
  }

  order_ = Storage(size_, scalar_type_, arena_);
  order_.fillOrderedData(0, 1, order_type_);

  if (strides.getSize() != size_) {
    strides_ = Storage(size_, scalar_type_, arena_);

    switch (scalar_type_) {
#define STRIDE_CASE(ST_TYPE, CPP_TYPE)                                         \
  case ScalarType::ST_TYPE:                                                    \
    darkside::fillStrides<CPP_TYPE>(strides_.getData(), shape_.getData(),      \
                                    order_.getData(), size_, arena_);          \
    break;

      STRIDE_CASE(INT_8, int8_t)
      STRIDE_CASE(INT_16, int16_t)
      STRIDE_CASE(INT_32, int32_t)
      STRIDE_CASE(INT_64, int64_t)
      STRIDE_CASE(UNSIGNED_INT_8, uint8_t)
      STRIDE_CASE(UNSIGNED_INT_16, uint16_t)
      STRIDE_CASE(UNSIGNED_INT_32, uint32_t)
      STRIDE_CASE(UNSIGNED_INT_64, uint64_t)
      STRIDE_CASE(FLOAT_32, float)
      STRIDE_CASE(FLOAT_64, double)

#undef STRIDE_CASE

    default:
      break;
    }
  } else {
    strides_ = strides;
    strides_.setScalarType(scalar_type_);
    strides_.setArena(arena_);
  }

  if (offsets.getSize() != size_) {
    offsets_ = Storage(size_, scalar_type_, arena_);
    offsets_.fillData(0);
  } else {
    offsets_ = offsets;
    offsets_.setScalarType(scalar_type_);
    offsets_.setArena(arena_);
  }
}

const Storage &Layout::getShape() const { return shape_; }
const Storage &Layout::getOrder() const { return order_; }
const Storage &Layout::getStrides() const { return strides_; }
const Storage &Layout::getOffsets() const { return offsets_; }
} // namespace startorch
