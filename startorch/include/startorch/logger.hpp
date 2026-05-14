#pragma once

#include "startorch/common.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

namespace startorch {
class Logger {
public:
  static void makeLog(LoggerType logger_type, const std::string &file, int line, const std::string &message) {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);

    std::cout << "[" << std::put_time(std::localtime(&time_t_now), "%H:%M:%S") << "] ";
    std::cout << "[" << getLoggerTypeString(logger_type) << "] ";
    std::cout << "[" << file << ":" << line << "] ";
    std::cout << message << std::endl;
  }

  static std::string getLoggerTypeString(LoggerType logger_type) {
    switch (logger_type) {
    case LoggerType::DEBUG:
      return "DEBUG";

    case LoggerType::INFO:
      return "INFO";

    case LoggerType::WARN:
      return "WARN";

    case LoggerType::ERROR:
      return "ERROR";

    default:
      return "UNKNOWN_LOGGER";
    }
  }

  static std::string getDeviceTypeString(DeviceType device_type) {
    switch (device_type) {
    case DeviceType::CPU:
      return "CPU: AMD Ryzen 5 5625U (12) @ 4.39 GHz";

    case DeviceType::GPU:
      return "GPU: NVIDIA GeForce RTX 3050 Mobile [Discrete]";

    default:
      return "UNKNOWN_DEVICE";
    }
  }

  static std::string getMemoryTypeString(MemoryType memory_type) {
    switch (memory_type) {
    case MemoryType::HOST:
      return "HOST";

    case MemoryType::DEVICE:
      return "DEVICE";

    case MemoryType::PINNED:
      return "PINNED";

    case MemoryType::UNIFIED:
      return "UNIFIED";

    default:
      return "UNKNOWN_MEMORY";
    }
  }
};
} // namespace startorch
