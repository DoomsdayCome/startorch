#pragma once

#include "startorch/common.hpp"

#include <cstdint>

namespace darkside {
template <typename T> struct CPPTypeToScalarType;

#define CPP_TYPE_TO_SCALAR_TYPE(T, N)                                                                                                                          \
  template <> struct CPPTypeToScalarType<T> {                                                                                                                  \
    static constexpr startorch::ScalarType getType = N;                                                                                                        \
  };

CPP_TYPE_TO_SCALAR_TYPE(void, startorch::ScalarType::UNKNOWN_SCALAR);
CPP_TYPE_TO_SCALAR_TYPE(float, startorch::ScalarType::FLOAT_32);
CPP_TYPE_TO_SCALAR_TYPE(double, startorch::ScalarType::FLOAT_64);
CPP_TYPE_TO_SCALAR_TYPE(int8_t, startorch::ScalarType::INT_8);
CPP_TYPE_TO_SCALAR_TYPE(int16_t, startorch::ScalarType::INT_16);
CPP_TYPE_TO_SCALAR_TYPE(int32_t, startorch::ScalarType::INT_32);
CPP_TYPE_TO_SCALAR_TYPE(int64_t, startorch::ScalarType::INT_64);
CPP_TYPE_TO_SCALAR_TYPE(uint8_t, startorch::ScalarType::UNSIGNED_INT_8);
CPP_TYPE_TO_SCALAR_TYPE(uint16_t, startorch::ScalarType::UNSIGNED_INT_16);
CPP_TYPE_TO_SCALAR_TYPE(uint32_t, startorch::ScalarType::UNSIGNED_INT_32);
CPP_TYPE_TO_SCALAR_TYPE(uint64_t, startorch::ScalarType::UNSIGNED_INT_64);

#undef CPP_TYPE_TO_SCALAR_TYPE

template <typename F> decltype(auto) ScalarTypeToCPPType(startorch::ScalarType scalar_type, F &&f) {
  switch (scalar_type) {
  case startorch::ScalarType::INT_8:
    return f(CPPTypeToScalarType<int8_t>{});

  case startorch::ScalarType::INT_16:
    return f(CPPTypeToScalarType<int16_t>{});

  case startorch::ScalarType::INT_32:
    return f(CPPTypeToScalarType<int32_t>{});

  case startorch::ScalarType::INT_64:
    return f(CPPTypeToScalarType<int64_t>{});

  case startorch::ScalarType::FLOAT_32:
    return f(CPPTypeToScalarType<float>{});

  case startorch::ScalarType::FLOAT_64:
    return f(CPPTypeToScalarType<double>{});

  case startorch::ScalarType::UNSIGNED_INT_8:
    return f(CPPTypeToScalarType<uint8_t>{});

  case startorch::ScalarType::UNSIGNED_INT_16:
    return f(CPPTypeToScalarType<uint16_t>{});

  case startorch::ScalarType::UNSIGNED_INT_32:
    return f(CPPTypeToScalarType<uint32_t>{});

  case startorch::ScalarType::UNSIGNED_INT_64:
    return f(CPPTypeToScalarType<uint64_t>{});

  default:
    // return f(CPPTypeToScalarType<void>{});
    break;
  }
}

uint64_t getScalarTypeSize(startorch::ScalarType scalar_type);
} // namespace darkside
