#pragma once

#include "startorch/device.hpp"

#include <cstdint>

namespace darkside {
void *makeMemory(uint64_t size, const startorch::Device &device);
void freeMemory(uint64_t pointer, const startorch::Device &device);
void copyMemory(void *destination, void *source,
                const startorch::DevicePair &device_pair);

class Buffer {
private:
  void *data_ = nullptr;
  uint64_t size_ = 0;
  startorch::Device device_ = startorch::Device();

public:
};
} // namespace darkside
