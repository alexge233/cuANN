#ifndef _cuANN_sums_HPP_
#define _cuANN_sums_HPP_
#include "includes.ihh"

namespace cuANN
{
__global__ void sums ( 
                       float * input,
                       float * output, 
                       unsigned int size 
                     )
{ 
    float sum = 0;

    for ( int i = 0; i < size; i++ )
    {   
        sum += input[threadIdx.x];
    }   
    
    // Sigmoid
    output[threadIdx.x] = 1.0 / (1.0 + exp ( -sum ) );

    // Fast Sigmoid
    //output[threadIdx.x] = sum / ( 1 + abs( sum ) );
}

}
#endif
