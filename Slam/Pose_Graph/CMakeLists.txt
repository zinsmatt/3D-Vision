cmake_minimum_required(VERSION 2.8)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Eigen
include_directories("/usr/include/eigen3")

# sophus 
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# g2o 
find_package(g2o REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

Find_Package(Ceres REQUIRED)


add_executable(pose_graph_g2o_SE3 pose_graph_g2o_SE3.cpp)
target_link_libraries(pose_graph_g2o_SE3
        g2o::core g2o::stuff g2o::types_sba g2o::opengl_helper
        ${CHOLMOD_LIBRARIES}
        )

add_executable(pose_graph_g2o_lie pose_graph_g2o_lie_algebra.cpp)
target_link_libraries(pose_graph_g2o_lie
        g2o::core g2o::stuff g2o::types_sba  g2o::opengl_helper
        ${CHOLMOD_LIBRARIES}
        Sophus::Sophus
        )


add_executable(pose_graph_ceres_lie pose_graph_ceres_lie.cpp)
target_link_libraries(pose_graph_ceres_lie
        ${CERES_LIBRARIES}
        Sophus::Sophus
)


add_executable(pose_graph_ceres_lie_autodiff pose_graph_ceres_lie_autodiff.cpp)
target_link_libraries(pose_graph_ceres_lie_autodiff
        ${CERES_LIBRARIES}
        Sophus::Sophus
)

        