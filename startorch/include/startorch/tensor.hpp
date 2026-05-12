#include "startorch/common.hpp"
#include "startorch/format.hpp"
#include "startorch/layout.hpp"
#include "startorch/memory.hpp"

#include <cstdint>
#include <initializer_list>
#include <memory>

namespace startorch {
class Tensor {
private:
  Layout layout_ = Layout();
  std::shared_ptr<Storage> shared_ = std::make_shared<Storage>();
  ScalarType scalar_type_ = ScalarType::FLOAT_32;
  Arena *arena_ = nullptr;
  uint64_t rank_;

public:
  Tensor(std::initializer_list<darkside::CPPValueToScalarValue> shape,
         std::initializer_list<darkside::CPPValueToScalarValue> storage,
         ScalarType scalar_type, Arena *arena);

  const Layout &getLayout() const;
};
} // namespace startorch
