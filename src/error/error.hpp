#ifndef _cuANN_error_HPP_
#define _cuANN_error_HPP_
#include "includes.ihh"

namespace cuANN
{
/// Calculate the Output Error: (Ideal[i] - Actual[i])^2
__global__ void squared_error ( 
                                float * ideal,
                                float * actual, 
                                float * errors
                            );

}
#endif
