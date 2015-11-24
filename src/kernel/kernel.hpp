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

/// Non-Zero predicate for thrust::count
/// Use it to find in vectors the amount of non-zero values
struct non_zero
{
    __host__ __device__ bool operator()( const float & x )
    {
      return x != 0.0;
    }
};

/** 
 * Sigmoid Activation Kernel: σ(x) = 1 / (1 + e^{-x} ).
 * @param input is an input device array,
 * @param size defines the size of the input array
 * @note this is a 1D grid kernel
 */
__global__ void sigmoid_activation ( float * input );

/**
 * @brief Calculate the Sigmoid Prime of: σ'(Σ[ji]) = σ(x) * (1 - σ(x))
 * @param sum_ji is the Σ[ji]
 * @param output store the result
 * @note this is a 1D grid kernel
 */
__global__ void sigmoid_prime (
                                float * sum_ji,
                                float * output
                              );

/** 
 * @brief Forward Propagation: `O[j] * W[i]`
 * @param weight is the Weights Matrix, in Row-Major format, where a row corresponds to a node's weights
 * @param input is a Column Vector, corresponding to `O[j]` (the output from previous nodes)
 * @param output is a Row-Major matrix which stores the result
 * @param w_size defines the Width of the Matrices (Weights & Output)
 * @note 2D kernel: X grid is Input/Row iterator, Y grid is Column iterator
 */
__global__ void forward_prop ( 
                               const float * weight, // W[ji] 
                               const float * input,  // O[j]
                               float * output,       // I[i]
                               unsigned int w_size   // weights per node (# of columns)
                             );

/** 
 * @brief Summarize Matrix Columns to a Row Vector: `Σ( O[j] * W[i] )`
 * @param w_mtx is  `O[j] * W[i]` from `forward_prop`
 * @param output is `Σ( O[j] * W[i] )` e.g., the Sumed columns as a Row
 * @param height defines the Height of the Matrix
 * @param width defines the Width of the Matrix
 * @note 1D grid: X is iterating Columns Index
 */
__global__ void sum_columns ( 
                                float * w_mtx,
                                float * output, 
                                unsigned int height,
                                unsigned int width
                            );

/**
 * @brief Delta Error of output layer: `-E * σ'( Σ(W[i]*O[j]) )`
 * @param sum is array of previous layer output dot incoming weights: `Σ ( W[i] * O[j])`
 * @param ideal is array `ideal` output, e.g., the Target output
 * @param actual is array `actual` output
 * @param delta is array of δ[i] of output layer
 * @param size is the amount of nodes in output layer
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
 * @brief Calculate Delta dot product of weights and next layer node Deltas: `W[ik] * δ[k]`
 * @note We multiply each weight, with its respective (next layer node's) δ[k]
 * @note `weights per node` must equal δ[k]
 * @note Y grid is `weights per node` as well as δ[k] size (since they are equal)
 * @note X grid is layer i nodes (could differ from k nodes)
 */
__global__ void delta_product (
                                float * w_ik,
                                float * d_k,
                                float * output,
                                unsigned int width
                              );

/**
 * @brief Sum Rows of Matrix: `Σ( W[ik] * δ[k] )`
 * @param w_ik_d is the matrix `W[ik] * δ[k]`
 * @param delta_i stores the Summed rows for each node in layer i
 * @param width defines the matrix width
 */
__global__ void delta_sum_rows (
                                float * w_ik_d,
                                float * delta_i,
                                unsigned int width
                               );

/** 
 * @brief Delta Error of a hidden layer:  `δ[i] = f'( Σ[ji] ) * Σ( W[ik] * δ[k])`
 * @param prime_ji is array `F'( Σ( W[ji] * N[j] ) )`
 * @param delta_i is array `δ[i]` the current layer's deltas (one for each neuron), our output
 * @warning delta_i **must** contain the values of `Σ( W[ik] * δ[k])`
 */
__global__ void delta_hidden (
                               float * prime_ji,
                               float * delta_i
                             );

/**
 * @brief Gradient Descent for all (but output) Neurons/Nodes: `∂E / ∂W[ik] = δ[k] * O[i]`
 * @note This is a per-layer calculation, not a network global function (not for the entire network) 
 * @param d_k is `δ[k]` the following layer's Node Delta
 * @param i_i is `O[i]` the output of the activation function from preceeding layer
 * @param g_ik is the product `∂E/∂W[ik]` e.g., the gradient (function output)
 * @param size_d defines the row width of the produced matrix
 * @note X grid is Node Delta count, Y grid is O[i] count 
 */
__global__ void gradient_descent (
                                    float * d_k,
                                    float * o_i,
                                    float * g_ik,
                                    unsigned int size_d
                                 );

/**
 * @brief Summarize Gradients for an entire Epoch 
 * @param gradient will be updates (the storage array)
 * @param new_value is the value to be added (the temp array)
 */
__global__ void sum_gradients (
                                float * gradient,
                                float * new_value
                              );    

/**
 * @brief Back-Propagation for all weights: `Δw(t) = ε * ( ∂E / ∂W[i] ) + α * ( Δw(t-1) )`
 * @param weights the entire network weights
 * @param gradients all gradients `∂E / ∂W[ik]` same size as weights
 * @param updates previous weight updates `Δw(t-1)` same size as weights, we set `Δw(t-1) = Δw(t)`
 * @param alpha the gradient momentum
 * @param epsilon the learning rate
 * @note this is a linear 1D grid, where X iterates weights, gradiients and updates (all same size)
 */
__global__ void back_prop (
                            float * weight,
                            float * gradient,
                            float * update,
                            float alpha,
                            float epsilon
                         );

/// Calculate the Output Error: (Ideal[i] - Actual[i])^2
__global__ void squared_error ( 
                                float * ideal,
                                float * actual, 
                                float * errors
                            );

};
#endif
