#pragma once

#include "startorch/device.hpp"

#include <cstdint>

namespace darkside {
template <typename T> void fillData(T *data, uint64_t size, T value, startorch::Device *device);
template <typename T> void fillIncreasedData(T *data, uint64_t size, T start, T step, startorch::Device *device);
template <typename T> void fillDecreasedData(T *data, uint64_t size, T start, T step, startorch::Device *device);

template <typename T, typename N>
void castData(T *destination, const N *source, uint64_t size, startorch::Device *destination_device, startorch::Device *source_device);

} // namespace darkside
