#pragma once

#include "startorch/common.hpp"
#include "startorch/memory.hpp"

#include <cstdint>

namespace startorch {
class Layout {
private:
  Storage shape_ = Storage();
  Storage order_ = Storage();
  Storage strides_ = Storage();
  Storage offsets_ = Storage();

public:
  Layout() = default;
  Layout(const Storage &shape, const Storage &order, const Storage &strides, const Storage &offsets);
  Layout(const Storage &shape, OrderType order_type, const Storage &strides, const Storage &offsets);

  Layout(const Layout &other) = default;
  Layout(Layout &&other) noexcept = default;

  ~Layout() = default;

  Layout &operator=(const Layout &other) = default;
  Layout &operator=(Layout &&other) noexcept = default;

  const Storage &getShape() const;
  const Storage &getOrder() const;
  const Storage &getStrides() const;
  const Storage &getOffsets() const;
  uint64_t getIndex(const Storage &indices) const;
};
} // namespace startorch
