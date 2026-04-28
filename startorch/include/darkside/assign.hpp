#pragma once

#include "startorch/device.hpp"

#include <cstdint>

namespace darkside {
template <typename T>
void fillData(void *data, uint64_t size, T value,
              const startorch::Device &device);
template <typename T>
void fillIncreaseData(void *data, uint64_t size,
                      const startorch::Device &device);
template <typename T>
void fillDecreaseData(void *data, uint64_t size,
                      const startorch::Device &device);
} // namespace darkside
