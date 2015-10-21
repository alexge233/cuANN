#include "prop.hpp"
namespace cuANN
{
__global__ void prop_vector ( 
                               float * weight, 
                               float * input, 
                               float * output, 
                               unsigned int size 
                            )
{ 
    float sum = 0;

    for ( int i = 0; i < size; i++ )
    {   
        sum += weight[threadIdx.x] * input[i];
    }   
    
    // Sigmoid
    output[threadIdx.x] = 1.0 / (1.0 + exp ( -sum ) );

    // Fast Sigmoid
    //output[threadIdx.x] = sum / ( 1 + abs( sum ) );
}

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
    if ( ( x < w_size ) && ( y < i_size ) )
    {
        output[y + x * i_size] = weight[x] * input[y];
    }
}

}
