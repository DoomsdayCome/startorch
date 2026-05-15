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

template <typename F> decltype(auto) ScalarTypeToCPPTypeNameSpace(startorch::ScalarType scalar_type, F &&f) {
  switch (scalar_type) {
  case startorch::ScalarType::INT_8:
    return f(int8_t{});

  case startorch::ScalarType::INT_16:
    return f(int16_t{});

  case startorch::ScalarType::INT_32:
    return f(int32_t{});

  case startorch::ScalarType::INT_64:
    return f(int64_t{});

  case startorch::ScalarType::FLOAT_32:
    return f(float{});

  case startorch::ScalarType::FLOAT_64:
    return f(double{});

  case startorch::ScalarType::UNSIGNED_INT_8:
    return f(uint8_t{});

  case startorch::ScalarType::UNSIGNED_INT_16:
    return f(uint16_t{});

  case startorch::ScalarType::UNSIGNED_INT_32:
    return f(uint32_t{});

  case startorch::ScalarType::UNSIGNED_INT_64:
    return f(uint64_t{});

  default:
    break;
  }
}
} // namespace darkside

namespace startorch {
using ElementVariant = std::variant<int8_t, int16_t, int32_t, int64_t, float, double, uint8_t, uint16_t, uint32_t, uint64_t>;

class Element {
private:
  ElementVariant value_ = 0;
  void *data_ = nullptr;
  ScalarType scalar_type_ = ScalarType::UNKNOWN_SCALAR;

public:
  Element() = default;

  Element(int8_t value) : value_(value), scalar_type_(startorch::ScalarType::INT_8) { data_ = &std::get<int8_t>(value_); }
  Element(int16_t value) : value_(value), scalar_type_(startorch::ScalarType::INT_16) { data_ = &std::get<int16_t>(value_); }
  Element(int32_t value) : value_(value), scalar_type_(startorch::ScalarType::INT_32) { data_ = &std::get<int32_t>(value_); }
  Element(int64_t value) : value_(value), scalar_type_(startorch::ScalarType::INT_64) { data_ = &std::get<int64_t>(value_); }
  Element(float value) : value_(value), scalar_type_(startorch::ScalarType::FLOAT_32) { data_ = &std::get<float>(value_); }
  Element(double value) : value_(value), scalar_type_(startorch::ScalarType::FLOAT_64) { data_ = &std::get<double>(value_); }
  Element(uint8_t value) : value_(value), scalar_type_(startorch::ScalarType::UNSIGNED_INT_8) { data_ = &std::get<uint8_t>(value_); }
  Element(uint16_t value) : value_(value), scalar_type_(startorch::ScalarType::UNSIGNED_INT_16) { data_ = &std::get<uint16_t>(value_); }
  Element(uint32_t value) : value_(value), scalar_type_(startorch::ScalarType::UNSIGNED_INT_32) { data_ = &std::get<uint32_t>(value_); }
  Element(uint64_t value) : value_(value), scalar_type_(startorch::ScalarType::UNSIGNED_INT_64) { data_ = &std::get<int64_t>(value_); }

  Element(const Element &other) = default;
  Element(Element &&other) noexcept = default;

  ~Element() = default;

  Element &operator=(const Element &other) = default;
  Element &operator=(Element &&other) noexcept = default;

  template <typename T> T *getData() { return (T *)data_; }
  template <typename T> const T *getData() const { return (T *)data_; }

  const void *getData() { return data_; }
  void *getData() const { return data_; }
  ScalarType getScalarType() const { return scalar_type_; }
};
} // namespace startorch
