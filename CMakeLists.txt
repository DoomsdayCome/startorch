cmake_minimum_required(VERSION 3.23)
project(test LANGUAGES CXX)

find_package(GTest REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(test
  "${CMAKE_CURRENT_SOURCE_DIR}/test/device.cpp" 
  "${CMAKE_CURRENT_SOURCE_DIR}/test/memory.cpp" 
)

target_include_directories(test PRIVATE 
  "${CMAKE_CURRENT_SOURCE_DIR}/startorch/include"
  "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(test 
  PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/startorch/library/libstartorch.so" 
    GTest::gtest 
    GTest::gtest_main
    CUDA::cudart
)
