cmake_minimum_required(VERSION 3.10)

project(pin_manage_memory)
# cuda
include_directories(SYSTEM $ENV{CUDA_PATH}/include)
set(CUDA_DIR $ENV{CUDA_HOME})
# set(CMAKE_CUDA_FLAGS "-I$ENV{CUDA_PATH}/include")
link_directories($ENV{CUDA_PATH}/lib64)
link_directories($ENV{CUDA_PATH}/lib64/stubs)

# enable_language(CUDA)
find_package(CUDA REQUIRED)

cuda_add_executable(pin_manage_memory pin_manage_memory.cu)
