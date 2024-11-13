cmake_minimum_required(VERSION 3.10)

project(argmax)
# cuda
include_directories(SYSTEM $ENV{CUDA_PATH}/include)
set(CUDA_DIR $ENV{CUDA_HOME})
# set(CMAKE_CUDA_FLAGS "-I$ENV{CUDA_PATH}/include")
link_directories($ENV{CUDA_PATH}/lib64)
link_directories($ENV{CUDA_PATH}/lib64/stubs)

# enable_language(CUDA)
find_package(CUDA REQUIRED)

cuda_add_executable(matrix_multiply_v2 matrix_multiply_v2.cu)
