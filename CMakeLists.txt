cmake_minimum_required(VERSION 3.0)
project(quadprog CXX)


find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "RelWithDebInfo")
add_library(quadprogpp_libs src/quadprogpp/QuadProg++.cc)
add_library(quadprog_eigen_libs src/quadprog_eigen/quadprog_eigen.cc)
add_library(eigen_qp_libs src/EigenQP/EigenQP.cc)

add_executable(quadprog_test main.cc)
target_link_libraries(quadprog_test PUBLIC Eigen3::Eigen quadprog_eigen_libs quadprogpp_libs eigen_qp_libs)

add_executable(speed_test speed_test.cc)
target_link_libraries(speed_test PUBLIC Eigen3::Eigen quadprog_eigen_libs quadprogpp_libs eigen_qp_libs)


###############################################################################
## packaging ##################################################################
###############################################################################

# all install commands get the same destination. this allows us to use paths
# relative to the executable.
#install(TARGETS example DESTINATION example_destination)
# this is basically a repeat of the file copy instruction that copies the
# resources in the build directory, but here we tell cmake that we want it
# in the package
#install(DIRECTORY resources DESTINATION example_destination)

# now comes everything we need, to create a package
# there are a lot more variables you can set, and some
# you need to set for some package types, but we want to
# be minimal here
set(CPACK_PACKAGE_NAME "QuadProg_Example")
set(CPACK_PACKAGE_VERSION "1.0.0")

# we don't want to split our program up into several things
set(CPACK_MONOLITHIC_INSTALL 1)

# This must be last
include(CPack)