cmake_minimum_required(VERSION 3.16)
project(ParaForge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PARAFORGE_BUILD_TESTS "Build unit tests" ON)
option(PARAFORGE_ENABLE_CUDA "Build the CUDA kernel, its tests and benchmark (needs nvcc)" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)  # benchmarks are meaningless at -O0
endif()

find_package(Threads REQUIRED)

add_library(paraforge src/matrix.cpp src/thread_pool.cpp)
target_include_directories(paraforge PUBLIC include)
target_link_libraries(paraforge PUBLIC Threads::Threads)
if(MSVC)
  target_compile_options(paraforge PRIVATE /W4)
else()
  target_compile_options(paraforge PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(paraforge_bench bench/matmul_bench.cpp)
target_link_libraries(paraforge_bench PRIVATE paraforge)

if(PARAFORGE_ENABLE_CUDA)
  if(CMAKE_VERSION VERSION_LESS 3.24)
    message(FATAL_ERROR "PARAFORGE_ENABLE_CUDA needs CMake >= 3.24 (CUDA_ARCHITECTURES native)")
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)  # the GPU(s) in this machine; override for other targets
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_library(paraforge_cuda src/matmul_cuda.cu)
  target_link_libraries(paraforge_cuda PUBLIC paraforge PRIVATE CUDA::cudart_static)

  add_executable(paraforge_cuda_bench bench/cuda_bench.cpp)
  target_link_libraries(paraforge_cuda_bench PRIVATE paraforge_cuda)
endif()

if(PARAFORGE_BUILD_TESTS)
  enable_testing()
  find_package(GTest CONFIG QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
      URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926)
    # Match the MSVC runtime of the parent project.
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  add_executable(paraforge_tests tests/matrix_test.cpp tests/matmul_parallel_test.cpp
    tests/thread_pool_test.cpp)
  target_link_libraries(paraforge_tests PRIVATE paraforge GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(paraforge_tests)

  if(PARAFORGE_ENABLE_CUDA)
    add_executable(paraforge_cuda_tests tests/matmul_cuda_test.cpp)
    target_link_libraries(paraforge_cuda_tests PRIVATE paraforge_cuda CUDA::cudart_static
      GTest::gtest_main)
    gtest_discover_tests(paraforge_cuda_tests)
  endif()
endif()
