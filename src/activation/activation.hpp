#ifndef _cuANN_activation_HPP_
#define _cuANN_activation_HPP_
#include "includes.ihh"
namespace cuANN
{

/// Sigmoid Activation Function
__host__ __device__ float sigmoid_func ( const float x );

/// Fast Sigmoid Activation Function
__host__ __device__ float fast_sigmoid ( const float x );

/// Hyperbolic Tangent
__host__ __device__ float tanh_func ( const float x );

}
#endif
