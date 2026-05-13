#include "startorch/layout.hpp"
#include "startorch/common.hpp"
#include "startorch/format.hpp"
#include "startorch/memory.hpp"

#include "darkside/kernel.cuh"

#include <cstdint>

namespace startorch {
#define STRIDE_CASE(scalar_type, cpp_type)                                     \
  case scalar_type:                                                            \
    darkside::fillStrides<cpp_type>(strides_.getData(), shape_.getData(),      \
                                    order_.getData(), size_, arena_);          \
    break;

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
  }

  shape_ = shape;
  shape_.setScalarType(scalar_type_);
  shape_.setArena(arena_);

  if (order.getSize() != size_) {
    order_ = Storage(size_, scalar_type_, arena_);
    order_.fillOrderedData(0, 1, order_type_);
  } else {
    order_ = order;
    order_.setScalarType(scalar_type_);

    void *data = nullptr;
    uint64_t byte = size_ * darkside::getScalarTypeSize(scalar_type_);

    Arena temp(byte, MemoryType::HOST, DeviceType::CPU);

    auto type = order_.getArena()->getDevice().getDeviceType();
    auto arena_type = arena_->getDevice().getDeviceType();

    if (type == DeviceType::CPU) {
      data = order_.getData();
    } else if (arena_type == DeviceType::CPU) {
      order_.setArena(arena_);
      data = order_.getData();
    } else {
      order_.setArena(&temp);
      data = order_.getData();
    }

    switch (scalar_type_) {

#define ORDER_CASE(scalar_type, cpp_type)                                      \
  case scalar_type: {                                                          \
    cpp_type *ptr = (cpp_type *)data;                                          \
                                                                               \
    bool row = true;                                                           \
    bool col = true;                                                           \
                                                                               \
    for (uint64_t i = 0; i < size_ - 1; i++) {                                 \
      if (ptr[i] > ptr[i + 1])                                                 \
        row = false;                                                           \
                                                                               \
      if (ptr[i] < ptr[i + 1])                                                 \
        col = false;                                                           \
    }                                                                          \
                                                                               \
    if (row)                                                                   \
      order_type_ = OrderType::ROW_MAJOR;                                      \
    else if (col)                                                              \
      order_type_ = OrderType::COLUMN_MAJOR;                                   \
    else                                                                       \
      order_type_ = OrderType::NONE_MAJOR;                                     \
                                                                               \
    break;                                                                     \
  }

      ORDER_CASE(ScalarType::INT_8, int8_t)
      ORDER_CASE(ScalarType::INT_16, int16_t)
      ORDER_CASE(ScalarType::INT_32, int32_t)
      ORDER_CASE(ScalarType::INT_64, int64_t)
      ORDER_CASE(ScalarType::UNSIGNED_INT_8, uint8_t)
      ORDER_CASE(ScalarType::UNSIGNED_INT_16, uint16_t)
      ORDER_CASE(ScalarType::UNSIGNED_INT_32, uint32_t)
      ORDER_CASE(ScalarType::UNSIGNED_INT_64, uint64_t)
      ORDER_CASE(ScalarType::FLOAT_32, float)
      ORDER_CASE(ScalarType::FLOAT_64, double)

#undef ORDER_CASE

    default:
      break;
    }

    order_.setArena(arena_);
  }

  if (strides.getSize() != size_) {
    strides_ = Storage(size_, scalar_type_, arena_);

    switch (scalar_type_) {

      STRIDE_CASE(ScalarType::INT_8, int8_t)
      STRIDE_CASE(ScalarType::INT_16, int16_t)
      STRIDE_CASE(ScalarType::INT_32, int32_t)
      STRIDE_CASE(ScalarType::INT_64, int64_t)
      STRIDE_CASE(ScalarType::UNSIGNED_INT_8, uint8_t)
      STRIDE_CASE(ScalarType::UNSIGNED_INT_16, uint16_t)
      STRIDE_CASE(ScalarType::UNSIGNED_INT_32, uint32_t)
      STRIDE_CASE(ScalarType::UNSIGNED_INT_64, uint64_t)
      STRIDE_CASE(ScalarType::FLOAT_32, float)
      STRIDE_CASE(ScalarType::FLOAT_64, double)

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
  }

  shape_ = shape;
  shape_.setScalarType(scalar_type_);
  shape_.setArena(arena_);

  order_ = Storage(size_, scalar_type_, arena_);
  order_.fillOrderedData(0, 1, order_type_);

  if (strides.getSize() != size_) {
    strides_ = Storage(size_, scalar_type_, arena_);

    switch (scalar_type_) {

      STRIDE_CASE(ScalarType::INT_8, int8_t)
      STRIDE_CASE(ScalarType::INT_16, int16_t)
      STRIDE_CASE(ScalarType::INT_32, int32_t)
      STRIDE_CASE(ScalarType::INT_64, int64_t)
      STRIDE_CASE(ScalarType::UNSIGNED_INT_8, uint8_t)
      STRIDE_CASE(ScalarType::UNSIGNED_INT_16, uint16_t)
      STRIDE_CASE(ScalarType::UNSIGNED_INT_32, uint32_t)
      STRIDE_CASE(ScalarType::UNSIGNED_INT_64, uint64_t)
      STRIDE_CASE(ScalarType::FLOAT_32, float)
      STRIDE_CASE(ScalarType::FLOAT_64, double)

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

#undef STRIDE_CASE

const Storage &Layout::getShape() const { return shape_; }
const Storage &Layout::getOrder() const { return order_; }
const Storage &Layout::getStrides() const { return strides_; }
const Storage &Layout::getOffsets() const { return offsets_; }

// uint64_t Layout::getIndex(const Storage &indices) const {
//   if (indices.getSize() != size_)
//     return 0;
//
//   Storage indices_ = indices;
//   indices_.setScalarType(scalar_type_);
//   indices_.setArena(arena_);
//
//
// }
} // namespace startorch
