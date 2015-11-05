#include "activation.hpp"

namespace cuANN
{

__global__ void sigmoid_kernel( float * input, unsigned int size )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x < size )
    {
        float value = input[x];
        input[x] = 1.f / (1.f + exp ( value ) );
    }
}

};
