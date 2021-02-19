cmake_minimum_required(VERSION 3.5)

project(NeuralNet_Layer_Cpp)

find_package(Boost 1.75 COMPONENTS unit_test_framework REQUIRED)
find_package(Armadillo 6.5 REQUIRED)

include_directories($(Boost_INCLUDE_DIR))
include_directories($(ARMADILLO_INCLUDE_DIR))

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(
    conv_test tests/test_conv_layer.cpp
)

target_link_libraries(conv_test LINK_PUBLIC ${Boost_LIBRARIES})
target_link_libraries(conv_test LINK_PUBLIC ${ARMADILLO_LIBRARIES})