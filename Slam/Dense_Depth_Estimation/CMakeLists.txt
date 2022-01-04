cmake_minimum_required(VERSION 2.8)
project(dense_monocular LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3")

############### dependencies ######################
# Eigen
# include_directories("/usr/include/eigen3")
find_package(Eigen3 3.3 REQUIRED)


# OpenCV
find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        Sophus::Sophus
        )
add_library(plot_tools plot.cpp)
target_link_libraries(plot_tools ${THIRD_PARTY_LIBS} Eigen3::Eigen)


add_executable(dense_mapping_custom dense_mapping_custom.cpp)
target_link_libraries(dense_mapping_custom ${THIRD_PARTY_LIBS} Eigen3::Eigen plot_tools)

add_executable(dense_mapping_custom_multithread dense_mapping_custom_multithread.cpp)
target_link_libraries(dense_mapping_custom_multithread ${THIRD_PARTY_LIBS} Eigen3::Eigen plot_tools)


add_executable(dense_mapping_custom_no_eigen dense_mapping_custom_no_eigen.cpp)
target_link_libraries(dense_mapping_custom_no_eigen ${THIRD_PARTY_LIBS} Eigen3::Eigen plot_tools)



add_executable(dense_mapping_custom_cuda
dense_mapping_custom_cuda.cpp
cuda_wrapper.cpp
kernels.cu
kernels.cuh)

target_link_libraries(dense_mapping_custom_cuda ${THIRD_PARTY_LIBS} Eigen3::Eigen plot_tools)
set_target_properties(dense_mapping_custom_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)