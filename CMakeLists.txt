cmake_minimum_required(VERSION 3.5)

project(NeuralNet_Layer_Cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

add_executable(
    main src/main.cpp
)