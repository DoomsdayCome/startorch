#pragma once

#include <cstdint>

constexpr uint64_t operator""_KiB(unsigned long long v) { return v << 10; }
constexpr uint64_t operator""_MiB(unsigned long long v) { return v << 20; }
constexpr uint64_t operator""_GiB(unsigned long long v) { return v << 30; }

namespace darkside {
inline constexpr uint64_t THREADS = 256;
inline constexpr uint64_t BLOCKS(uint64_t size) { return (size + THREADS - 1) / THREADS; }
} // namespace darkside

namespace startorch {
enum class DeviceType : uint8_t {
  UNKNOWN_DEVICE = 0,
  CPU = 1,
  GPU = 2,
};

inline constexpr DeviceType CPU = DeviceType::CPU;
inline constexpr DeviceType GPU = DeviceType::GPU;

enum class MemoryType : uint8_t {
  UNKNOWN_MEMORY = 0,
  HOST = 1,
  DEVICE = 2,
  PINNED = 3,
  UNIFIED = 4,
};

inline constexpr MemoryType HOST = MemoryType::HOST;
inline constexpr MemoryType DEVICE = MemoryType::DEVICE;
inline constexpr MemoryType PINNED = MemoryType::PINNED;
inline constexpr MemoryType UNIFIED = MemoryType::UNIFIED;

enum class ScalarType : uint8_t {
  UNKNOWN_SCALAR = 0,
  FLOAT_8 = 1,
  FLOAT_16 = 2,
  FLOAT_32 = 3,
  FLOAT_64 = 4,
  INT_8 = 5,
  INT_16 = 6,
  INT_32 = 7,
  INT_64 = 8,
  UNSIGNED_INT_8 = 9,
  UNSIGNED_INT_16 = 10,
  UNSIGNED_INT_32 = 11,
  UNSIGNED_INT_64 = 12,
};

inline constexpr ScalarType FLOAT_8 = ScalarType::FLOAT_8;
inline constexpr ScalarType FLOAT_16 = ScalarType::FLOAT_16;
inline constexpr ScalarType FLOAT_32 = ScalarType::FLOAT_32;
inline constexpr ScalarType FLOAT_64 = ScalarType::FLOAT_64;
inline constexpr ScalarType INT_8 = ScalarType::INT_8;
inline constexpr ScalarType INT_16 = ScalarType::INT_16;
inline constexpr ScalarType INT_32 = ScalarType::INT_32;
inline constexpr ScalarType INT_64 = ScalarType::INT_64;
inline constexpr ScalarType UNSIGNED_INT_8 = ScalarType::UNSIGNED_INT_8;
inline constexpr ScalarType UNSIGNED_INT_16 = ScalarType::UNSIGNED_INT_16;
inline constexpr ScalarType UNSIGNED_INT_32 = ScalarType::UNSIGNED_INT_32;
inline constexpr ScalarType UNSIGNED_INT_64 = ScalarType::UNSIGNED_INT_64;

enum class LoggerType : uint8_t {
  UNKNOWN_LOGGER = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
};

inline constexpr LoggerType DEBUG = LoggerType::DEBUG;
inline constexpr LoggerType INFO = LoggerType::INFO;
inline constexpr LoggerType WARN = LoggerType::WARN;
inline constexpr LoggerType ERROR = LoggerType::ERROR;

enum class OrderType : uint8_t {
  UNKNOWN_MAJOR = 0,
  ROW_MAJOR = 1,
  COLUMN_MAJOR = 2,
};

inline constexpr OrderType ROW_MAJOR = OrderType::ROW_MAJOR;
inline constexpr OrderType COLUMN_MAJOR = OrderType::COLUMN_MAJOR;

enum class OwnerType {
  UNKNOWN_OWNER = 0,
  REFERENCE = 1,
  OWNED = 2,
};

inline constexpr OwnerType REFERENCE = OwnerType::REFERENCE;
inline constexpr OwnerType OWNED = OwnerType::OWNED;
} // namespace startorch
