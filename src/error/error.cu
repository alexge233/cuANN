#include "error.hpp"
namespace cuANN
{

__global__ void squared_error ( 
                                float * ideal,
                                float * actual, 
                                float * errors
                            )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    float diff = ideal[x] - actual[x];
    errors[x] = diff * diff;
}

}
