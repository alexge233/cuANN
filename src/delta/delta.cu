#include "delta.hpp"
#include <stdio.h>
namespace cuANN
{

__global__ void delta_output (
                                float * sum,
                                float * ideal,
                                float * actual,
                                float * delta,
                                unsigned int index
                             )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    float error = -1.f * ( actual[x] - ideal[x]);
    float f = sigmoid_prime( sum[x+index] );
    delta[x+index] = error * f;
}

__global__ void delta_hidden (
                               const float * w_sum,
                               unsigned int w_sum_size,
                               const float * delta_k,
                               unsigned int delta_k_size,
                               float * delta_i,
                               unsigned int delta_i_size,
                               const float * weight_i,
                               unsigned int weight_i_size
                             )
{
    // TODO: think how this can be done in a single kernel
    //       because Im afraid it may require two kernels
}

__host__ __device__ float sigmoid_prime( const float & input )
{
    float out = 1.f / (1.0 + exp ( input ) );
    return ( out * ( 1.f - out ) );
}


};
