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
find_package(Boost COMPONENTS system thread serialization REQUIRED)
find_package(Threads REQUIRED)

# What does this line do ?
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)

# ?
set(CUDA_SEPERABLE_COMPILATION OFF)
set(Boost_DEBUG OFF)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)

# nvcc flags & C++ Flags - NOTE Optimisation can go here
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};#-gencode arch=compute_20,code=sm_20
                                       #-gencode arch=compute_20,code=sm_21;
                                       -gencode arch=compute_30,code=sm_30;
                                       -std=c++11;
                                       #-DVERBOSE;
                                       -DNDEBUG;
                                       -O3;
                                       -Xcompiler -fPIC;
                                       --use_fast_math;
                                       )


add_subdirectory(${SRC})
add_subdirectory(${SAMPLES})

# Build the library
cuda_add_library(cuANN
                ${SRC}/ann/ann.cu
                ${SRC}/data/data.cu
                ${SRC}/kernel/kernel.cu
                ${SRC}/trainer/trainer_data.cu
                )

# Link with Boost and pthreads - This must be TESTED
target_link_libraries(cuANN ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ${Boost_SERIALIZATION_LIBRARY})

# Then we may build the samples

# XOR with sigmoid bipolar
cuda_compile(XOR_O ${SAMPLES}/xor.cu )
cuda_add_executable(xor_sigmoid ${XOR_O})
target_link_libraries(xor_sigmoid cuANN)

# XOR with normal tanh
cuda_compile(XOR_TANH_O ${SAMPLES}/xor_tanh.cu )
cuda_add_executable(xor_tanh ${XOR_TANH_O})
target_link_libraries(xor_tanh cuANN)

# XOR with scaled tanh
cuda_compile(XOR_TANH_SCALED_O ${SAMPLES}/xor_tanh_scaled.cu )
cuda_add_executable(xor_tanh_scaled ${XOR_TANH_SCALED_O})
target_link_libraries(xor_tanh_scaled cuANN)

# Load XOR tanh scaled from file
cuda_compile(LOAD_XOR_O ${SAMPLES}/load_xor.cu )
cuda_add_executable(load_xor_net ${LOAD_XOR_O})
target_link_libraries(load_xor_net cuANN)

# Abelone Sample
cuda_compile(ABEL_O ${SAMPLES}/abelone.cu)
cuda_add_executable(abelone ${ABEL_O})
target_link_libraries(abelone cuANN)

# Diabetes Sample
cuda_compile(DIA_O ${SAMPLES}/diabetes.cu)
cuda_add_executable(diabetes ${DIA_O})
target_link_libraries(diabetes cuANN)

# Gene Sample
cuda_compile(GENE_O ${SAMPLES}/gene.cu)
cuda_add_executable(gene ${GENE_O})
target_link_libraries(gene cuANN)

# Mushroom Sample
cuda_compile(MUSHROOM_O ${SAMPLES}/mushroom.cu)
cuda_add_executable(mushroom ${MUSHROOM_O})
target_link_libraries(mushroom cuANN)

# Robot Sample
cuda_compile(ROBOT_O ${SAMPLES}/robot.cu)
cuda_add_executable(robot ${ROBOT_O})
target_link_libraries(robot cuANN)

# Soybean Sample
cuda_compile(SOYBEAN_O ${SAMPLES}/soybean.cu)
cuda_add_executable(soybean ${SOYBEAN_O})
target_link_libraries(soybean cuANN)

# Thyroid Sample
cuda_compile(THYROID_O ${SAMPLES}/thyroid.cu)
cuda_add_executable(thyroid ${THYROID_O})
target_link_libraries(thyroid cuANN)

# TODO: define the install location
