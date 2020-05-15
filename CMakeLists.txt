cmake_minimum_required(VERSION 3.16)
project(smpl_prog)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)

find_package(MPI REQUIRED)
# add this line only when you are using openmpi which has a different c++ bindings
add_definitions(-DOMPI_SKIP_MPICXX)

find_package(OpenMP)


add_executable(smpl_prog main.cpp)

target_link_libraries(smpl_prog mpi pthread OpenMP::OpenMP_CXX)