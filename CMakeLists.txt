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
find_package(Boost COMPONENTS system thread REQUIRED)
find_package(Threads REQUIRED)

# What does this line do ?
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)

# ?
set(CUDA_SEPERABLE_COMPILATION ON)

# nvcc flags & C++ Flags - NOTE Optimisation can go here
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};#-gencode arch=compute_20,code=sm_20
                                       #-gencode arch=compute_20,code=sm_21;
                                       -gencode arch=compute_30,code=sm_30;
                                       -std=c++11;
                                       -DVERBOSE;
                                       #-DTHRUST_DEBUG;
                                       #-DNDEBUG;
                                       #-O3;
                                       -Xcompiler -fPIC;
                                       --use_fast_math)

add_subdirectory(${SRC})
add_subdirectory(${SAMPLES})

# Build the library
cuda_add_library(cuANN
                ${SRC}/ann/ann.cu
                ${SRC}/data/data.cu
                ${SRC}/kernel/kernel.cu
                ${SRC}/trainer/trainer.cu
                ${SRC}/trainer/trainer_data.cu
                ${SRC}/trainer/trainer_pool.cu
                )

# Link with Boost and pthreads - This must be TESTED
target_link_libraries(cuANN ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})

# Then we may build the samples

# the XOR Sample
cuda_compile(XOR_O ${SAMPLES}/xor.cu )
cuda_add_executable(xor ${XOR_O})
target_link_libraries(xor cuANN)

# the Testing Sample
cuda_compile(TEST_O ${SAMPLES}/test.cu)
cuda_add_executable(test_net ${TEST_O})
target_link_libraries(test_net cuANN)

# Abelone Sample
cuda_compile(ABEL_O ${SAMPLES}/abelone.cu)
cuda_add_executable(abelone ${ABEL_O})
target_link_libraries(abelone cuANN)

# and Finally, define the install location
