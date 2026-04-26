#pragma once

#include <cstdint>

namespace startorch {
enum class DeviceType : uint8_t {
  CPU = 0,
  GPU = 1,
};

inline constexpr DeviceType CPU = DeviceType::CPU;
inline constexpr DeviceType GPU = DeviceType::GPU;

enum class MemoryType : uint8_t {
  DEFAULT = 0,
  PINNED = 1,
  UNIFIED = 2,
};

inline constexpr MemoryType DEFAULT = MemoryType::DEFAULT;
inline constexpr MemoryType PINNED = MemoryType::PINNED;
inline constexpr MemoryType UNIFIED = MemoryType::UNIFIED;

// enum class ScalarType : uint8_t {
//   INT_8 = 0,
//   INT_16 = 1,
//   INT_32 = 2,
//   INT_64 = 3,
//   FLOAT_8 = 4,
//   FLOAT_16 = 5,
//   FLOAT_32 = 6,
//   FLOAT_64 = 7,
//   UNSIGNED_INT_8 = 8,
//   UNSIGNED_INT_16 = 9,
//   UNSIGNED_INT_32 = 10,
//   UNSIGNED_INT_64 = 11,
// };

// const ScalarType INT_8 = ScalarType::INT_8;
// const ScalarType INT_16 = ScalarType::INT_16;
// const ScalarType INT_32 = ScalarType::INT_32;
// const ScalarType INT_64 = ScalarType::INT_64;
// const ScalarType FLOAT_8 = ScalarType::FLOAT_8;
// const ScalarType FLOAT_16 = ScalarType::FLOAT_16;
// const ScalarType FLOAT_32 = ScalarType::FLOAT_32;
// const ScalarType FLOAT_64 = ScalarType::FLOAT_64;
// const ScalarType UNSIGNED_INT_8 = ScalarType::UNSIGNED_INT_8;
// const ScalarType UNSIGNED_INT_16 = ScalarType::UNSIGNED_INT_16;
// const ScalarType UNSIGNED_INT_32 = ScalarType::UNSIGNED_INT_32;
// const ScalarType UNSIGNED_INT_64 = ScalarType::UNSIGNED_INT_64;
} // namespace startorch
