#pragma once

#include "startorch/common.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

namespace startorch {
class Logger {
public:
  static void makeLog(LoggerType logger_type, const std::string &file, int line,
                  const std::string &message) {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);

    // Format: [TIME] [LEVEL] [FILE:LINE] MESSAGE
    std::cout << "[" << std::put_time(std::localtime(&time_t_now), "%H:%M:%S")
              << "] ";
    std::cout << "[" << getLoggerTypeString(logger_type) << "] ";
    std::cout << "[" << file << ":" << line << "] ";
    std::cout << message << std::endl;
  }

private:
  static std::string getLoggerTypeString(LoggerType level) {
    switch (level) {
    case LoggerType::DEBUG:
      return "DEBUG";

    case LoggerType::INFO:
      return "INFO";

    case LoggerType::WARN:
      return "WARN";

    case LoggerType::ERROR:
      return "ERROR";

    default:
      return "UNKNOWN";
    }
  }
};
} // namespace startorch
