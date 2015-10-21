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
                    /usr/include
                    )

# Ignore warnings from includes
include_directories(SYSTEM ${INCLUDE_HEADERS})
include_directories(${SRC}/)

# Use custom Modules
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

# Older versions of NVCC do not support shared libs
# Build Shared (so) unless you want a static version for some reason
set(BUILD_SHARED_LIBS ON)

# packages
find_package(CUDA 6.5 REQUIRED )
find_package(Thrust 1.7.2 REQUIRED)

# What does this line do ?
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)

# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};#-gencode arch=compute_20,code=sm_20
                                       #-gencode arch=compute_20,code=sm_21;
                                       -gencode arch=compute_30,code=sm_30;
                                       -std=c++11; 
                                       --use_fast_math)

add_subdirectory(${SRC})
add_subdirectory(${SAMPLES})

# Build the library
cuda_add_library(cuANN
                ${SRC}/ann/ann.cu
                ${SRC}/prg/prg.cu
                ${SRC}/sigmoid/sigmoid.cu
                ${SRC}/prop/prop.cu
                )

# Then we may build the samples

# The commands below will compile a cuda executable
# Use them to compile the SAMPLES
cuda_compile(MAIN_O ${SAMPLES}/main.cu)
#cuda_compile(DRIVER_O driver.cu OPTIONS -G)

cuda_add_executable(main ${MAIN_O})
target_link_libraries(main cuANN)
# and Finally, define the install location
