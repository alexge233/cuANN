#ifndef _cuANN_kernel_HPP_
#define _cuANN_kernel_HPP_
#include "includes.ihh"
namespace cuANN
{

/// Pseudo-Random Number Generator
struct prg
{
    float min, max;
    unsigned int seed; 

    __host__ __device__ prg( float _a, float _b, unsigned int _s ) 
    : min(_a), max(_b), seed( _s )
    {};

    __host__ __device__ float operator()( int idx ) const
    {
        thrust::default_random_engine rng ( seed );
        //thrust::minstd_rand rng;
        thrust::uniform_real_distribution<float> dist(min, max);
        rng.discard(idx);
        return dist(rng);
    }
};


/** 
 * Sigmoid Activation Kernel: f(x) = 1 / (1 + e^{-x} ).
 * @param input is an input device array,
 * @param size defines the size of the input array
 */
__global__ void sigmoid_kernel( float * input, unsigned int size );

/** 
 * @brief Layer propagation: vector * matrix dot product
 * @note requires a 2D grid
 * @warning the result is a Matrix, where each row represents the output from multiplying one input with all its weights
 *          the matrix is vectorised using thrust.
 *          the format is: Input[i]*Weight[i]
 */
__global__ void prop_kernel ( 
                              const float * weight, 
                              const float * input, 
                              float * output, 
                              unsigned int w_size
                            );

/// @brief Summarize columns into a row vector
__global__ void sum_columns ( 
                                float * w_mtx,
                                float * output, 
                                unsigned int w_size
                            );

/**
 * @brief Delta Error of last (output) layer (used for Gradient Descent)
 * @param sum is array `Sum ( Weight * Input)`
 * @param ideal is array `ideal` output
 * @param actual is array `actual` output
 * @param delta is array of `-E * f'( S(W*I) )`
 * @param size is the amount of nodes (neurons) in output layer
 * @note all parameters (w_sum,out_err,delta) should be the same size
 */
__global__ void delta_output (
                                float * sum,
                                float * ideal,
                                float * actual,
                                float * delta,
                                unsigned int index
                             );


/** 
 * @brief Delta Error of a hidden layer (used for Gradient Descent)
 * @param f_ji is array `F'( Sum( W[ji] * N[j] ) )`: sigmoid_derivative
 * @param delta_i is array `δ[i]` the current layer's deltas (one for each neuron), our output
 * @param weight_ik outgoing weights from this layer i, to layer k
 * @param size_i defines size of this layer [i], e.g., the neuron count
 * @param size_w size of weights per neuron/node
 *
 * Formula for calculating the Delta Errors in Hidden layers is:
 *      δ[i] = f'(S[i]) * Σ(w[ik] * δ[k])
 *
 * This is a three-step calculation, but we already have the S[i] stored from forward-propagation.
 */
__global__ void delta_hidden (
                               float * sum_ji,
                               float * delta_i,
                               float * delta_k,
                               float * weight_ik,
                               unsigned int size_i,
                               unsigned int size_w
                             );


/// Calculate the Output Error: (Ideal[i] - Actual[i])^2
__global__ void squared_error ( 
                                float * ideal,
                                float * actual, 
                                float * errors
                            );

};
#endif
