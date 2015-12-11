# cuANN
## Artificial Neural Network for CUDA
### Version 0.2

A simple Feed Forward Multilayered Neural Network, using CUDA and Thrust.
I started working on this project, after I realised that (at the time of writting this) no
simple CUDA-based ANN library exists for C++.
Some that do exist use Python, rely on closed-source libraries, are obscure, have been abandoned or are CLI programs, rather than libraries.

cuANN is written in a minimal way, and templated where possible, in order for you to change and adapt it to your needs.
It is inspired by FANN, which has for a long time been the most widely used ANN library.
However, thanks to GPGPU, cuANN aspires to help you use large Neural Networks on your NVIDIA GPU.

This version (0.2) is written with CUDA Compute 3.0 in mind, and is tested on a GTX660.
It has not been optimized yet, and most of the cycles are spent on `cudaMemcpy` and `cudaMalloc`.
Most of the samples when run on my GTX660 use on average 25% to 35%.
Next version (0.3) will hopefully be better optimised.

### Building
In order to build:

    mkdir build
    cd build
    cmake ..
    make -j8

The dependencies are:

```
    Boost >= 1.49
    CUDA >= 6.5
    Thrust >= 1.7
    CMake >= 2.8
    g++   >= 4.8
```

*cuANN uses C++11 features*, so make sure you have a modern enough compiler (g++ or clang).
If you wish to change the CUDA Compute to an earlier version, do so by editing the `CMakeLists.txt` file.

### Examples
You can browse the *simple exclusive* or (XOR) example under `samples/`, or the more complex *Abelone*.
The data used to train them is the same as with libFANN.

### NOTE: CUDA Compute compatability.
I have tested cuANN using CUDA compute 3.0.
Newer versions and GPUs should work fine, however I can't guarantee it will work for older cards.
**WARNING** if you wish to use an older GPU, change the `CMakeLists.txt` and use the correct `-gencode` flag for your card's architecture and sm code.

### NOTE: Neural Networks using *Back-Propagation* are bizzare machines: they may fail to learn, or may take too long to learn.
This greatly depends on the network architecture (input nodes, hidden nodes, hidden layers, etc) as well as the learning rate *epsilon* and the momentum *alpha*.
