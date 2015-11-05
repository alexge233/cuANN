#ifndef _cuANN_delta_HPP_
#define _cuANN_delta_HPP_
#include "includes.ihh"
namespace cuANN
{

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
 * @param w_sum is array `Sum( W[ji] * O[j] )`
 * @param w_sum_size is size of w_sums
 * @param delta_k is array `δ[κ]` the previous layer's deltas (one for each neuron)
 * @param delta_k_size is size of delta_k
 * @param delta_i is array `δ[i]` the current layer's deltas (one for each neuron), our output values
 * @param delta_i_size is size of delta_i
 * @param weight_i is array `w[i]` the weights from layer [i] to [k].
 * @param weight_i_size is size of weight_i
 *
 * Formula for calculating the Delta Errors in Hidden layers is:
 *  S[i] = Σ(w[ji] * o[j]). where w[ji] = weight from previous to current layer, o[j] = previous neuron output
 *  δ[i] = f'(S[i]) * Σ(w[ik] * δ[k]). where w[ik] = weight from current to next layer, δ[k] = next layer node delta
 *
 * This is a three-step calculation, but we already have the S[i] stored from forward-propagation.
 * We only compute the Sum: Σ(w[ik] * δ[k])
 * and we also compute the derivative output: f'(S[i])
 */
__global__ void delta_hidden (
                               const float * w_sum,
                               unsigned int w_sum_size,
                               const float * delta_k,
                               unsigned int delta_k_size,
                               float * delta_i,
                               unsigned int delta_i_size,
                               const float * weight_i,
                               unsigned int weight_i_size
                             );

/** 
 * Sigmoid Derivate/Prime Function: f(x)' = f(x) * ( 1 - f(x) ).
 * @param input is a single float value
 * @return the derivative value
 */
__host__ __device__ float sigmoid_prime ( const float & input );


};
#endif
