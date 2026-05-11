cmake_minimum_required(VERSION 3.20)
project(llama_lowlatency_runtime CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(third_party/llama.cpp)

add_executable(llama_latency src/main.cpp)

target_link_libraries(llama_latency PRIVATE llama)

