# cuANN
## Artificial Neural Network for CUDA
### alpha version

A simple Feed Forward Multilayered Neural Network, using CUDA and Thrust.
I started working on this project, after I realised that (at the time of writting this) no
simple CUDA-based ANN library exists for C++.
Some that do exist use Python, rely on closed-source libraries, are obscure, have been abandoned or are CLI programs, rather than libraries.

cuANN is written in a minimalistic way, as I'm learning how Neural Networks work, I'm writting this library.
It is inspired by FANN, which has for a long time been the most widely used ANN library.
However, thanks to GPGPU, cuANN aspires to help you use large Neural Networks on your NVIDIA GPU.

This version (0.1) is written with CUDA Compute 3.0 in mind, and is tested on a GTX660.

### Building
In order to build:

    mkdir build
    cd build
    cmake ..
    make -j8

The dependencies are:

    CUDA >= 6.5
    Thrust >= 1.7
    CMake >= 2.8
    g++   >= 4.8

*cuANN uses C++11 features*, so make sure you have a modern enough compiler (g++ or clang).
If you wish to change the CUDA Compute to an earlier version, do so by editing the `CMakeLists.txt` file.

### Examples
You can browse the *simple exclusive* or (XOR) example under `samples/`, or the more complex *Abelone*.
The data used to train them is the same as with libFANN.

## NOTE: Current implementation is not using enough of the GPU available computability.
