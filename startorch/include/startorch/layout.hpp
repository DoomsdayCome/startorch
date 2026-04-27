#include "startorch/common.hpp"
#include "startorch/memory.hpp"

namespace startorch {
class Layout {
private:
  ScalarType scalar_type_ = ScalarType::UNSIGNED_INT_64;
  Device device_ = Device();

  Storage shape_ = Storage(0, scalar_type_, device_);
  Storage order_ = Storage(0, scalar_type_, device_);
  Storage strides_ = Storage(0, scalar_type_, device_);
  Storage offsets_ = Storage(0, scalar_type_, device_);

public:
  Layout() = default;
  Layout(const Storage &shape, const Storage &order, const Storage &strides,
         const Storage &offsets, const Device &device);

  Layout(const Layout &other) = default;
  Layout(Layout &&other) noexcept = default;

  ~Layout() = default;

  Layout &operator=(const Layout &other) = default;
  Layout &operator=(Layout &&other) noexcept = default;

  const Storage &getShape() const;
  const Storage &getOrder() const;
  const Storage &getStrides() const;
  const Storage &getOffsets() const;
};
} // namespace startorch
