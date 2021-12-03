cmake_minimum_required(VERSION 2.8)
project(direct_method_gn)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 ${SSE_FLAGS} -g -O3 -march=native")

find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

add_executable(direct_method_gn direct_method_gn.cpp)
target_link_libraries(direct_method_gn ${OpenCV_LIBS} Sophus::Sophus)
