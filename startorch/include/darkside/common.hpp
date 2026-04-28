#pragma once

#include <cstdint>

namespace darkside {
inline constexpr uint64_t THREADS = 256;
#define BLOCKS(size) (((size) + THREADS - 1) / THREADS)
}
