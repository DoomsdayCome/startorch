#pragma once

#include "startorch/common.hpp"

#include <cstdint>
#include <variant>

namespace darkside {
template <typename T> struct CPPTypeToScalarType;

#define CPP_TYPE_TO_SCALAR_TYPE(cpp_type, scalar_type)                                                                                                         \
  template <> struct CPPTypeToScalarType<cpp_type> {                                                                                                           \
    static constexpr startorch::ScalarType getType = scalar_type;                                                                                              \
  };

CPP_TYPE_TO_SCALAR_TYPE(int8_t, startorch::ScalarType::INT_8);
CPP_TYPE_TO_SCALAR_TYPE(int16_t, startorch::ScalarType::INT_16);
CPP_TYPE_TO_SCALAR_TYPE(int32_t, startorch::ScalarType::INT_32);
CPP_TYPE_TO_SCALAR_TYPE(int64_t, startorch::ScalarType::INT_64);
CPP_TYPE_TO_SCALAR_TYPE(float, startorch::ScalarType::FLOAT_32);
CPP_TYPE_TO_SCALAR_TYPE(double, startorch::ScalarType::INT_64);
CPP_TYPE_TO_SCALAR_TYPE(uint8_t, startorch::ScalarType::UNSIGNED_INT_8);
CPP_TYPE_TO_SCALAR_TYPE(uint16_t, startorch::ScalarType::UNSIGNED_INT_16);
CPP_TYPE_TO_SCALAR_TYPE(uint32_t, startorch::ScalarType::UNSIGNED_INT_32);
CPP_TYPE_TO_SCALAR_TYPE(uint64_t, startorch::ScalarType::UNSIGNED_INT_64);

#undef CPP_TYPE_TO_SCALAR_TYPE

template <startorch::ScalarType S> struct ScalarTypeToCPPType;

#define SCALAR_TYPE_TO_CPP_TYPE(scalar_type, cpp_type)                                                                                                         \
  template <> struct ScalarTypeToCPPType<scalar_type> {                                                                                                        \
    using getType = cpp_type;                                                                                                                                  \
  };

SCALAR_TYPE_TO_CPP_TYPE(startorch::ScalarType::INT_8, int8_t);
SCALAR_TYPE_TO_CPP_TYPE(startorch::ScalarType::INT_16, int16_t);
SCALAR_TYPE_TO_CPP_TYPE(startorch::ScalarType::INT_32, int32_t);
SCALAR_TYPE_TO_CPP_TYPE(startorch::ScalarType::INT_64, int64_t);
SCALAR_TYPE_TO_CPP_TYPE(startorch::ScalarType::FLOAT_32, float);
SCALAR_TYPE_TO_CPP_TYPE(startorch::ScalarType::FLOAT_64, double);
SCALAR_TYPE_TO_CPP_TYPE(startorch::ScalarType::UNSIGNED_INT_8, uint8_t);
SCALAR_TYPE_TO_CPP_TYPE(startorch::ScalarType::UNSIGNED_INT_16, uint16_t);
SCALAR_TYPE_TO_CPP_TYPE(startorch::ScalarType::UNSIGNED_INT_32, uint32_t);
SCALAR_TYPE_TO_CPP_TYPE(startorch::ScalarType::UNSIGNED_INT_64, uint64_t);

#undef SCALAR_TYPE_TO_CPP_TYPE

inline constexpr uint64_t getScalarTypeSize(startorch::ScalarType scalar_type) {
  switch (scalar_type) {
  case startorch::ScalarType::INT_8:
  case startorch::ScalarType::UNSIGNED_INT_8:
    return 1;

  case startorch::ScalarType::INT_16:
  case startorch::ScalarType::UNSIGNED_INT_16:
    return 2;

  case startorch::ScalarType::INT_32:
  case startorch::ScalarType::UNSIGNED_INT_32:
    return 4;

  case startorch::ScalarType::INT_64:
  case startorch::ScalarType::UNSIGNED_INT_64:
    return 8;

  case startorch::ScalarType::FLOAT_32:
    return sizeof(float);

  case startorch::ScalarType::FLOAT_64:
    return sizeof(double);

  default:
    return 0;
  }
}
} // namespace darkside

namespace startorch {
using ElementVariant = std::variant<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, double, void *>;

class Element {
private:
  ElementVariant data_ = nullptr;
  startorch::ScalarType scalar_type_ = startorch::ScalarType::UNKNOWN_SCALAR;

public:
  Element() = default;

  Element(int8_t data) : data_(data), scalar_type_(startorch::ScalarType::INT_8) {}
  Element(int16_t data) : data_(data), scalar_type_(startorch::ScalarType::INT_16) {}
  Element(int32_t data) : data_(data), scalar_type_(startorch::ScalarType::INT_32) {}
  Element(int64_t data) : data_(data), scalar_type_(startorch::ScalarType::INT_64) {}
  Element(float data) : data_(data), scalar_type_(startorch::ScalarType::FLOAT_32) {}
  Element(double data) : data_(data), scalar_type_(startorch::ScalarType::FLOAT_64) {}
  Element(uint8_t data) : data_(data), scalar_type_(startorch::ScalarType::UNSIGNED_INT_8) {}
  Element(uint16_t data) : data_(data), scalar_type_(startorch::ScalarType::UNSIGNED_INT_16) {}
  Element(uint32_t data) : data_(data), scalar_type_(startorch::ScalarType::UNSIGNED_INT_32) {}
  Element(uint64_t data) : data_(data), scalar_type_(startorch::ScalarType::UNSIGNED_INT_64) {}

  Element(const Element &other) = default;
  Element(Element &&other) noexcept = default;

  ~Element() = default;

  Element &operator=(const Element &other) = default;
  Element &operator=(Element &&other) noexcept = default;

  template <typename T> T getData() const { return static_cast<T>(data_); };

  ElementVariant getData() const { return data_; }
  startorch::ScalarType getScalarType() const { return scalar_type_; }
};
} // namespace startorch
