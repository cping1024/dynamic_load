cmake_minimum_required(VERSION 2.8)
project(dynamic_load CXX)

set(CMAEK_C_FLGAS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

add_subdirectory(src)
add_subdirectory(sample)
