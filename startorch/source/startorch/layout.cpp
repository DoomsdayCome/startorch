#include "startorch/layout.hpp"
#include "startorch/common.hpp"
#include "startorch/memory.hpp"

#include "darkside/kernel.cuh"

namespace startorch {
Layout::Layout(const Storage &shape, const Storage &order,
               const Storage &strides, const Storage &offsets, Arena *arena)
    : arena_(arena), size_(shape.getSize()) {

  if (size_ == 0) {
    shape_.setArena(arena_);
    order_.setArena(arena_);
    strides_.setArena(arena_);
    offsets_.setArena(arena_);
    return;
  }

  if (arena_ == nullptr)
    return;

  shape_ = shape;
  shape_.setScalarType(scalar_type_);
  shape_.setArena(arena_);

  if (order.getSize() != size_) {
    order_ = Storage(size_, scalar_type_, arena_);
    order_.fillOrderedData(0, 1, order_type_);
  } else {
    order_ = order;
    order_.setScalarType(scalar_type_);
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

Layout::Layout(uint64_t size, Arena *arena)
    : size_(size), arena_(arena), shape_(size, scalar_type_, arena),
      order_(size, scalar_type_, arena), strides_(size, scalar_type_, arena),
      offsets_(size, scalar_type_, arena) {

  shape_.fillData(size);
  order_.fillOrderedData(0, 1, order_type_);
  offsets_.fillData(0);

  if (size_ > 0) {
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
  }
}

const Storage &Layout::getShape() const { return shape_; }
const Storage &Layout::getOrder() const { return order_; }
const Storage &Layout::getStrides() const { return strides_; }
const Storage &Layout::getOffsets() const { return offsets_; }
} // namespace startorch
