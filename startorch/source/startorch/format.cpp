#include "startorch/format.hpp"
#include "startorch/common.hpp"

#include <cstdint>

namespace startorch {
uint64_t getScalarTypeSize(startorch::ScalarType scalar_type) {
  switch (scalar_type) {
  case ScalarType::INT_8:
  case ScalarType::UNSIGNED_INT_8:
    return 1;

  case ScalarType::INT_16:
  case ScalarType::UNSIGNED_INT_16:
    return 2;

  case ScalarType::INT_32:
  case ScalarType::UNSIGNED_INT_32:
    return 4;

  case ScalarType::INT_64:
  case ScalarType::UNSIGNED_INT_64:
    return 8;

  case ScalarType::FLOAT_32:
    return sizeof(float);

  case ScalarType::FLOAT_64:
    return sizeof(double);

  default:
    return 0;
  }
}
} // namespace startorch
