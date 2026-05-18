cmake_minimum_required(VERSION 3.23)
project(startorch LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

find_package(CUDAToolkit REQUIRED)

add_library(startorch SHARED
  source/startorch/device.cpp
  source/startorch/memory.cpp
  source/startorch/layout.cpp

  source/darkside/kernel.cu
)

set_target_properties(startorch PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/library"
  CUDA_ARCHITECTURES native
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(startorch
  PUBLIC  include
  PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(startorch
  PRIVATE CUDA::cudart_static
)
