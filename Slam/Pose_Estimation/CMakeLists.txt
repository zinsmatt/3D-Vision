cmake_minimum_required(VERSION 2.8)
project(vo1)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV 3 REQUIRED)
find_package(g2o REQUIRED)
find_package(Sophus REQUIRED)

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})



include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)




add_executable(pose_estimation_3d2d_gauss_newton pose_estimation_3d2d_gauss_newton.cpp)
target_link_libraries(pose_estimation_3d2d_gauss_newton
        g2o::core
        g2o::stuff
        Sophus::Sophus
        ${OpenCV_LIBS})


add_executable(pose_estimation_3d2d_g2o pose_estimation_3d2d_g2o.cpp)
target_link_libraries(pose_estimation_3d2d_g2o
        g2o::core
        g2o::stuff
        Sophus::Sophus
        ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d_ceres_quat pose_estimation_3d2d_ceres_quat.cpp)
target_link_libraries(pose_estimation_3d2d_ceres_quat
        ${CERES_LIBRARIES}
        Sophus::Sophus
        ${OpenCV_LIBS})


add_executable(pose_estimation_3d2d_ceres_sophus pose_estimation_3d2d_ceres_sophus.cpp)
target_link_libraries(pose_estimation_3d2d_ceres_sophus
        ${CERES_LIBRARIES}
        Sophus::Sophus
        ${OpenCV_LIBS})

add_executable(pose_estimation_3d2d_ceres_sophus_analytical pose_estimation_3d2d_ceres_sophus_analytical.cpp)
target_link_libraries(pose_estimation_3d2d_ceres_sophus_analytical
        ${CERES_LIBRARIES}
        Sophus::Sophus
        ${OpenCV_LIBS})


add_executable(pose_estimation_3d3d_icp pose_estimation_3d3d_icp.cpp)
target_link_libraries(pose_estimation_3d3d_icp
        g2o::core
        g2o::stuff
        Sophus::Sophus
        ${OpenCV_LIBS})

add_executable(pose_estimation_3d3d_icp_ceres pose_estimation_3d3d_icp_ceres.cpp)
target_link_libraries(pose_estimation_3d3d_icp_ceres
        ${CERES_LIBRARIES}
        Sophus::Sophus
        ${OpenCV_LIBS})
