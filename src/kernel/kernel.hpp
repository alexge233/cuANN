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
 * @note this is a 1D grid kernel
 */
__global__ void sigmoid_activation( 
                                    float * input, 
                                    unsigned int size 
                                  );

/**
 * @brief Calculate the Sigmoid Derivative of: f'(Σ[ji])
 * @param sum_ji is the Σ[ji]
 * @param output store the result
 * @note this is a 1D grid kernel
 */
__global__ void sigmoid_derivative(
                                    float * sum_ji,
                                    float * output
                                  );

/** 
 * @brief Layer propagation: vector * matrix dot product
 * @note this is a 2D grid kernel
 * @warning the result is a Matrix, where each row represents the output from multiplying one input with all its weights
 *          the matrix is vectorised using thrust.
 *          the format is: Input[i]*Weight[i]
 */
__global__ void forward_prop ( 
                               const float * weight, // w[ji] 
                               const float * input,  // a[j]
                               float * output,       // a[i]
                               unsigned int w_size   // weights per node
                             );

/** 
 * @brief Summarize columns into a row vector
 * @param w_mtx
 * @param output
 * @param w_size
 * @note this is a 1D grid kernel
 */
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
 * @brief Calculate `W[ik] * δ[k]`
 * @note We multiply each weight, with its respective δ[k]
 * @note `weights per node` must equal δ[k] - else it makes no sense
 * @note Y grid is `weights per node` as well as δ[k] size (since they are equal)
 * @note X grid is layer i nodes (could differ from k nodes)
 */
__global__ void delta_product (
                                float * w_ik,
                                float * d_k,
                                float * output,
                                unsigned int w_size
                              );

/** 
 * @brief Delta Error of a hidden layer:  `δ[i] = f'( Σ[ji] ) * Σ( W[ik] * δ[k])`
 * @param prime_ji is array `F'( Σ( W[ji] * N[j] ) )`
 * @param delta_i is array `δ[i]` the current layer's deltas (one for each neuron), our output
 * @warning delta_i already contains the values of `Σ( W[ik] * δ[k])`
 */
__global__ void delta_hidden (
                               float * prime_ji,
                               float * delta_i
                             );


/// Calculate the Output Error: (Ideal[i] - Actual[i])^2
__global__ void squared_error ( 
                                float * ideal,
                                float * actual, 
                                float * errors
                            );

};
#endif
