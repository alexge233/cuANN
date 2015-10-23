#include "activation.hpp"

namespace cuANN
{

 __host__ __device__ float sigmoid_func ( const float x )
{
    return 1.0 / (1.0 + exp ( -x ) );
}

__host__ __device__ float fast_sigmoid ( const float x )
{
    return x / ( 1 + abs( x ) );
}

__host__ __device__ float tanh_func ( const float x )
{
    return tanh( x );
}

__global__ void sigmoid_kernel( float * input, unsigned int size )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x < size )
    {
        float value = input[x];
        input[x] = 1.0 / (1.0 + exp ( value ) );
    }
}

}
