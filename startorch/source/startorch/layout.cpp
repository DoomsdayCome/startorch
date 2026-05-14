#include "startorch/layout.hpp"
#include "startorch/common.hpp"
#include "startorch/format.hpp"
#include "startorch/memory.hpp"

#include "darkside/kernel.cuh"

#include <cstdint>

namespace startorch {
Layout::Layout(const Storage &shape, const Storage &order, const Storage &strides, const Storage &offsets) {
  if (shape.getSize() == 0)
    return;
}

const Storage &Layout::getShape() const { return shape_; }
const Storage &Layout::getOrder() const { return order_; }
const Storage &Layout::getStrides() const { return strides_; }
const Storage &Layout::getOffsets() const { return offsets_; }

} // namespace startorch
