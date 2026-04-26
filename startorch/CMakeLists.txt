cmake_minimum_required(VERSION 3.23)

project(startorch LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

add_library(startorch STATIC)

target_sources(startorch
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/source/startorch/device.cpp
)

target_include_directories(startorch
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(startorch
  PRIVATE
    CUDA::cudart
)

set_target_properties(startorch PROPERTIES
    CUDA_ARCHITECTURES native
)

set_target_properties(startorch PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/library
)
