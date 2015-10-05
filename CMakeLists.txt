# CMakeLists.txt for cuANN 
project(cuANN)

set (cuANN_VERSION_MAJOR 0)
set (cuANN_VERSION_MINOR 1)
set (SRC ${SRC} src)
set (SAMPLES ${SAMPLES} samples)

# required cmake version
cmake_minimum_required(VERSION 2.8)

# Lib path
set(LIBRARY_PATH ${LIBRARY_PATH}
                 /lib
                 /usr/lib
                 /usr/lib64)

# Include path
set(INCLUDE_HEADERS ${INCLUDE_HEADERS}
                    /usr/local/include
                    /usr/include)

# Ignore warnings from includes
include_directories(SYSTEM ${INCLUDE_HEADERS})

# Use custom Modules
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

# Older versions of NVCC do not support shared libs
set(BUILD_SHARED_LIBS OFF)

# packages
find_package(CUDA REQUIRED)
find_package(Thrust REQUIRED)

# What does this line do ?
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)

# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20
                                       -gencode arch=compute_20,code=sm_21
                                       -gencode arch=compute_21,code=sm_21)

add_subdirectory(${SRC})
add_subdirectory(${SAMPLES})

# The commands below will compile a cuda executable
#cuda_compile(HELPER_O helper.cu)
#cuda_compile(DRIVER_O driver.cu OPTIONS -G)
#cuda_add_executable(driver ${HELPER_O} ${DRIVER_O})

# But we want to create a Library first!
cuda_add_library(cuANN
                ${SRC}/ann/ann.cu
                )

# Then we may build the samples

# and Finally, define the install location
