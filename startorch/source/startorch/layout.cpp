#include "startorch/layout.hpp"
#include "startorch/common.hpp"
#include "startorch/memory.hpp"

#include "darkside/assign.cuh"

namespace startorch {
Layout::Layout(const Storage &shape, const Storage &order,
               const Storage &strides, const Storage &offsets, Arena *arena) {
  if (shape.getSize() == 0) {
    *this = Layout();
    arena_ = arena;

    return;
  }

  if (shape.getScalarType() != scalar_type_) {
    
  }

}

Layout::Layout(uint64_t size, Arena *arena)
    : size_(size), arena_(arena) {
  shape_ = Storage(size_, scalar_type_, arena_);
  order_ = Storage(size_, scalar_type_, arena_);
  strides_ = Storage(size_, scalar_type_, arena_);
  offsets_ = Storage(size_, scalar_type_, arena_);

  shape_.fillData(size);
  order_.fillIncreasedData(0, 1);
  offsets_.fillData(0);

  switch (scalar_type_) {
  case ScalarType::INT_8:
    darkside::fillStrides<int8_t>(strides_.getData(), shape_.getData(),
                                  order_.getData(), size_, arena_);
    break;

  case ScalarType::INT_16:
    darkside::fillStrides<int16_t>(strides_.getData(), shape_.getData(),
                                   order_.getData(), size_, arena_);
    break;

  case ScalarType::INT_32:
    darkside::fillStrides<int32_t>(strides_.getData(), shape_.getData(),
                                   order_.getData(), size_, arena_);
    break;

  case ScalarType::INT_64:
    darkside::fillStrides<int64_t>(strides_.getData(), shape_.getData(),
                                   order_.getData(), size_, arena_);
    break;

  case ScalarType::UNSIGNED_INT_8:
    darkside::fillStrides<uint8_t>(strides_.getData(), shape_.getData(),
                                   order_.getData(), size_, arena_);
    break;

  case ScalarType::UNSIGNED_INT_16:
    darkside::fillStrides<uint16_t>(strides_.getData(), shape_.getData(),
                                    order_.getData(), size_, arena_);
    break;

  case ScalarType::UNSIGNED_INT_32:
    darkside::fillStrides<uint32_t>(strides_.getData(), shape_.getData(),
                                    order_.getData(), size_, arena_);
    break;

  case ScalarType::UNSIGNED_INT_64:
    darkside::fillStrides<uint64_t>(strides_.getData(), shape_.getData(),
                                    order_.getData(), size_, arena_);
    break;

  case ScalarType::FLOAT_32:
    darkside::fillStrides<float>(strides_.getData(), shape_.getData(),
                                 order_.getData(), size_, arena_);
    break;

  case ScalarType::FLOAT_64:
    darkside::fillStrides<double>(strides_.getData(), shape_.getData(),
                                  order_.getData(), size_, arena_);
    break;

  default:
    break;
  }
}

const Storage &Layout::getShape() const { return shape_; }
const Storage &Layout::getOrder() const { return order_; }
const Storage &Layout::getStrides() const { return strides_; }
const Storage &Layout::getOffsets() const { return offsets_; }
} // namespace startorch
