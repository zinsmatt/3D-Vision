cmake_minimum_required(VERSION 2.8)
project(ch6)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# g2o
find_package(PythonLibs 3.6)

find_package(g2o REQUIRED
             COMPONENTS
             g2o::core
             g2o::stuff
             g2o::types_sba
             g2o::types_sim3
             g2o::solver_dense
             g2o::solver_eigen
             g2o::solver_csparse
             g2o::csparse_extension
             g2o::types_slam2d
             g2o::types_slam3d
             OPTIONAL_COMPONENTS
             g2o::csparse)


# Eigen
include_directories("/usr/include/eigen3")




add_executable(steepestDescent steepest_descent.cpp)
target_include_directories(steepestDescent SYSTEM PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    "/home/mzins/lib/matplotlib-cpp"
)
target_link_libraries(steepestDescent 
${OpenCV_LIBS}
${PYTHON_LIBRARIES}
)



add_executable(newton newton.cpp)
target_include_directories(newton SYSTEM PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    "/home/mzins/lib/matplotlib-cpp"
)
target_link_libraries(newton 
${OpenCV_LIBS}
${PYTHON_LIBRARIES}
)


add_executable(gaussNewton gauss_newton.cpp)
target_include_directories(gaussNewton SYSTEM PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    "/home/mzins/lib/matplotlib-cpp"
)
target_link_libraries(gaussNewton 
${OpenCV_LIBS}
${PYTHON_LIBRARIES}
)

add_executable(curve_fitting_ceres curve_fitting_ceres.cpp)
target_include_directories(curve_fitting_ceres SYSTEM PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    "/home/mzins/lib/matplotlib-cpp"
)
target_link_libraries(curve_fitting_ceres 
${OpenCV_LIBS}
${PYTHON_LIBRARIES}
${CERES_LIBRARIES}
)

add_executable(curve_fitting_g2o curve_fitting_g2o.cpp)
target_include_directories(curve_fitting_g2o SYSTEM PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    "/home/mzins/lib/matplotlib-cpp"
)
target_link_libraries(curve_fitting_g2o 
${OpenCV_LIBS}
${PYTHON_LIBRARIES}
g2o::core
g2o::stuff
g2o::types_sba
g2o::types_sim3
g2o::solver_dense
g2o::solver_eigen
g2o::solver_csparse
g2o::csparse_extension
g2o::types_slam2d
g2o::types_slam3d
)
