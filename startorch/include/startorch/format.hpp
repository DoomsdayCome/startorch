#pragma once

#include "startorch/common.hpp"

#include <cstdint>
#include <type_traits>

namespace darkside {
template <typename T> struct CPPTypeToScalar;

#define CPP_TO_SCALAR(cpp_type, scalar_type)                                   \
  template <> struct CPPTypeToScalar<cpp_type> {                               \
    static constexpr startorch::ScalarType type = scalar_type;                 \
  };

CPP_TO_SCALAR(int8_t, startorch::ScalarType::INT_8);
CPP_TO_SCALAR(int16_t, startorch::ScalarType::INT_16);
CPP_TO_SCALAR(int32_t, startorch::ScalarType::INT_32);
CPP_TO_SCALAR(int64_t, startorch::ScalarType::INT_64);
CPP_TO_SCALAR(float, startorch::ScalarType::FLOAT_32);
CPP_TO_SCALAR(double, startorch::ScalarType::INT_64);
CPP_TO_SCALAR(uint8_t, startorch::ScalarType::UNSIGNED_INT_8);
CPP_TO_SCALAR(uint16_t, startorch::ScalarType::UNSIGNED_INT_16);
CPP_TO_SCALAR(uint32_t, startorch::ScalarType::UNSIGNED_INT_32);
CPP_TO_SCALAR(uint64_t, startorch::ScalarType::UNSIGNED_INT_64);

#undef CPP_TO_SCALAR

template <startorch::ScalarType S> struct ScalarTypeToCPP;

#define SCALAR_TO_CPP(scalar_type, cpp_type)                                   \
  template <> struct ScalarTypeToCPP<scalar_type> {                            \
    using type = cpp_type;                                                     \
  };

SCALAR_TO_CPP(startorch::ScalarType::INT_8, int8_t);
SCALAR_TO_CPP(startorch::ScalarType::INT_16, int16_t);
SCALAR_TO_CPP(startorch::ScalarType::INT_32, int32_t);
SCALAR_TO_CPP(startorch::ScalarType::INT_64, int64_t);
SCALAR_TO_CPP(startorch::ScalarType::FLOAT_32, float);
SCALAR_TO_CPP(startorch::ScalarType::FLOAT_64, double);
SCALAR_TO_CPP(startorch::ScalarType::UNSIGNED_INT_8, uint8_t);
SCALAR_TO_CPP(startorch::ScalarType::UNSIGNED_INT_16, uint16_t);
SCALAR_TO_CPP(startorch::ScalarType::UNSIGNED_INT_32, uint32_t);
SCALAR_TO_CPP(startorch::ScalarType::UNSIGNED_INT_64, uint64_t);

#undef SCALAR_TO_CPP

class ScalarValueToCPP {
private:
  startorch::ScalarType scalar_type_ = startorch::ScalarType::UNSIGNED_INT_64;

  union {
    int64_t i_;
    double d_;
    uint64_t u_{0};
  };

public:
  ScalarValueToCPP() = default;

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  ScalarValueToCPP(T v) {
    if constexpr (std::is_signed_v<T>) {
      scalar_type_ = startorch::ScalarType::INT_64;
      i_ = static_cast<int64_t>(v);
    } else {
      scalar_type_ = startorch::ScalarType::UNSIGNED_INT_64;
      u_ = static_cast<uint64_t>(v);
    }
  }

  ScalarValueToCPP(double v) {
    scalar_type_ = startorch::ScalarType::FLOAT_64;
    d_ = v;
  }

  template <typename T> T value() const {
    switch (scalar_type_) {
    case startorch::ScalarType::INT_64:
      return static_cast<T>(i_);

    case startorch::ScalarType::UNSIGNED_INT_64:
      return static_cast<T>(u_);

    case startorch::ScalarType::FLOAT_64:
      return static_cast<T>(d_);

    default:
      return static_cast<T>(u_);
    }
  }
};

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
