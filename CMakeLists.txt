cmake_minimum_required(VERSION 3.09)
project(
  yacx-ocal-comparison
  VERSION 0.0.1
  LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

include_directories(include)
include_directories(include/yacx)
include_directories(include/ocal)

link_libraries(cuda nvrtc)

add_executable(sumarray src/sumarray.cpp)
