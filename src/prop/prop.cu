#include "prop.hpp"
namespace cuANN
{

__global__ void prop_matrix ( 
                              float * weight, 
                              float * input, 
                              float * output, 
                              unsigned int w_size, 
                              unsigned int i_size 
                            )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // ensure that thread x and y is within bounds
    if ( ( x < i_size ) && ( y < w_size ) )
    {
        // Input i, is at Grid X
        float in_i = input[x];

        // Weight for Input i, is at grid X * weight size + Y index
        // where Y indexes the weights.
        float w_i = weight[w_size * x + y];

        // Output is a simple multiplication
        output[w_size * x + y] = in_i * w_i;
    }
}

}
