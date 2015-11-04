#ifndef _cuANN_activation_HPP_
#define _cuANN_activation_HPP_
#include "includes.ihh"
namespace cuANN
{

/** 
 * Sigmoid Activation Kernel: f(x) = 1 / (1 + e^{-x} ).
 * @param input is an input device array,
 * @param size defines the size of the input array
 */
__global__ void sigmoid_kernel( float * input, unsigned int size );

/** 
 * Sigmoid Derivate/Prime Function: f(x)' = f(x) * ( 1 - f(x) ).
 * @param input is a single float value
 * @return the derivative value
 */
__host__ __device__ float sigmoid_prime ( const float & input );

// TODO: I could also implement the tanh_kernel and tanh_prime for usage with TANH instead of SIGMOID

}
#endif
