# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(wave-sdk-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(wave STATIC src/wave.cpp)
target_include_directories(wave PUBLIC include)

add_executable(test_vector_add tests/test_vector_add.cpp)
target_link_libraries(test_vector_add PRIVATE wave)
