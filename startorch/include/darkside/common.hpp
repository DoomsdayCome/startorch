#pragma once

#include <cstdint>

namespace darkside {
inline constexpr uint64_t THREADS = 256;

inline constexpr uint64_t BLOCKS(uint64_t size) {
  return (size + THREADS - 1) / THREADS;
}
}
