#ifndef _cuANN_sigmoid_HPP_
#define _cuANN_sigmoid_HPP_
#include "includes.ihh"
namespace cuANN
{

/// Sigmoid Activation Function
 __host__ __device__ float sigmoid ( const float x );

/// Fast Sigmoid Activation Function
__host__ __device__ float fast_sigmoid ( const float x );

}
#endif
