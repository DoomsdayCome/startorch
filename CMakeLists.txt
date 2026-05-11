cmake_minimum_required(VERSION 3.23)
project(test LANGUAGES CXX)

find_package(GTest REQUIRED)

add_executable(test
    "test/device.cpp" 
)

target_include_directories(test PRIVATE 
  "${CMAKE_CURRENT_SOURCE_DIR}/startorch/include"
)

target_link_libraries(test 
    PRIVATE 
        "${CMAKE_CURRENT_SOURCE_DIR}/startorch/library/libstartorch.so" 
        GTest::gtest 
        GTest::gtest_main
)
