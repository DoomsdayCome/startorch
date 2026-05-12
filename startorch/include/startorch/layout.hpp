#pragma once

#include "startorch/common.hpp"
#include "startorch/memory.hpp"

#include <cstdint>

namespace startorch {
class Layout {
private:
  ScalarType scalar_type_ = ScalarType::UNSIGNED_INT_64;
  Arena *arena_ = nullptr;
  OrderType order_type_ = OrderType::ROW_MAJOR;
  uint64_t size_ = 0;
  Storage shape_ = Storage(size_, scalar_type_, arena_);
  Storage order_ = Storage(size_, scalar_type_, arena_);
  Storage strides_ = Storage(size_, scalar_type_, arena_);
  Storage offsets_ = Storage(size_, scalar_type_, arena_);

public:
  Layout() = default;
  Layout(const Storage &shape, const Storage &order, const Storage &strides,
         const Storage &offsets, Arena *arena);
  Layout(const Storage &shape, OrderType order_type, const Storage &strides,
         const Storage &offsets, Arena *arena);

  Layout(const Layout &other) = default;
  Layout(Layout &&other) noexcept = default;

  ~Layout() = default;

  Layout &operator=(const Layout &other) = default;
  Layout &operator=(Layout &&other) noexcept = default;

  const Storage &getShape() const;
  const Storage &getOrder() const;
  const Storage &getStrides() const;
  const Storage &getOffsets() const;
  uint64_t getIndex() const;
};
} // namespace startorch
