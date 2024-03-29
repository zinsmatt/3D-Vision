cmake_minimum_required(VERSION 2.8)

project(bundle_adjustment)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

Find_Package(Eigen3 REQUIRED)
find_package(g2o REQUIRED)
Find_Package(Ceres REQUIRED)
Find_Package(Sophus REQUIRED)
Find_Package(CSparse REQUIRED)


include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})

add_library(bal_common common.cpp)
add_executable(bundle_adjustment_ceres bundle_adjustment_ceres.cpp)
add_executable(bundle_adjustment_g2o_numerical_diff bundle_adjustment_g2o_numerical_diff.cpp)
add_executable(bundle_adjustment_g2o_analytical bundle_adjustment_g2o_analytical.cpp)
add_executable(bundle_adjustment_g2o_analytical_no_intrinsics bundle_adjustment_g2o_analytical_no_intrinsics.cpp)
add_executable(bundle_adjustment_g2o_autodiff bundle_adjustment_g2o_autodiff.cpp)

target_link_libraries(bundle_adjustment_ceres ${CERES_LIBRARIES} Sophus::Sophus bal_common)
target_link_libraries(bundle_adjustment_g2o_numerical_diff g2o::core g2o::stuff g2o::csparse_extension cxsparse Sophus::Sophus bal_common)
target_link_libraries(bundle_adjustment_g2o_analytical g2o::core g2o::stuff g2o::csparse_extension cxsparse Sophus::Sophus bal_common)
target_link_libraries(bundle_adjustment_g2o_analytical_no_intrinsics g2o::core g2o::stuff g2o::csparse_extension cxsparse Sophus::Sophus bal_common)
target_link_libraries(bundle_adjustment_g2o_autodiff ${CERES_LIBRARIES} g2o::core g2o::stuff g2o::csparse_extension cxsparse Sophus::Sophus bal_common) # need ceres just for the rotation functions
