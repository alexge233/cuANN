#include "sums.hpp"

namespace cuANN
{
__global__ void sum_columns ( 
                                float * input,
                                float * output, 
                                unsigned int w_size,
                                unsigned int i_size
                            )
{
    //NOTE - This work with a 1D Grid
    // X thread iterates the vectorised matrix input
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    float total;
    for ( int i = 0; i < i_size; i++ )
    {
        total += input[ (i * w_size) + x];
    }
    output[x] = total;
}

}
