#ifndef _cuANN_kernel_HPP_
#define _cuANN_kernel_HPP_
#include "includes.ihh"
namespace cuANN
{

/// Signum function, 1 if > 0, 0 if 0, -1 if < 0
__device__ static float sgn(const float x)
{
    return (0.f < x) - (x < 0.f);
}  

/// Sigmoid: `σ(x) = 1 / 1 + e^( -x )`
struct sigmoid
{
    sigmoid()=default;
    __device__ float operator()(const float x) const
    {
        float denom = __fadd_rz(1.f,__expf(-x));
        return __fdividef(1.f,denom);
    }
};

/// Sigmoid Derivative: `σ'(x) = σ(x) * (1 - σ(x) )`
struct sigmoid_deriv
{
    sigmoid_deriv()=default;
    __device__ float operator()(const float x) const
    {
        float denom = __fadd_rz(1.f,__expf(-x));
        float sig= __fdividef(1.f,denom);
        return __fmul_rz(sig,(1.f-sig));
    }
};

/// Sigmoid Bipolar: `σ(x) = -1 + 2 / (1 + e^-x)`
struct sigmoid_bipolar
{
    __device__ float operator()(const float x) const
    {
        float denom = __fadd_rz(1.f,__expf(-x));
        return __fdividef(1,denom);
    }
};

/// Sigmoid Bipolar Derivative: `σ(x) = 0.5 * (1 + σ(x)) * (1 – σ(x) )`
struct sigmoid_bipolar_deriv
{
    sigmoid_bipolar_deriv()=default;
    __device__ float operator()(const float x) const
    {
        float denom = __fadd_rz(1,__expf(-x));
        float sig= __fdividef(1,denom);
        float rhs = 1-sig;
        float lhs = __fadd_rz(1,sig);
        float inner= __fmul_rz(lhs,rhs);
        return __fmul_rz(0.5,inner);
    }
};

/// Normal Hyperbolic Tangent: `tanh(x) = e^(x) - e^(-x) / e^(x) + e^(-x)i`
struct tanh_norm
{
    tanh_norm()=default;
    __device__ float operator()(const float x) const
    {
        return tanhf(x);
    }
};

/// Normal Hyperbolic Tangent Derivative: `1 / cosh^2(x)`
struct tanh_norm_deriv
{
    tanh_norm_deriv()=default;
    __device__ float operator()(const float x) const
    {
        float cosh_x = coshf(x);
        return __fdividef(1,__fmul_rz(cosh_x,cosh_x));
    }
};

/// Scaled Hyperbolic Tangent: `1.7159 * tanh(2.f/3.f*x)`
/// @note: The function is scaled in range {-1,1} to avoid learning saturation
struct tanh_scaled
{
    tanh_scaled()=default;
    __device__ float operator()(const float x) const
    {
        float value = __fmul_rz(0.666666667,x);
        float tanh_vl=  tanhf(value);
        return __fmul_rz(1.7159,tanh_vl);
    }
};

/// Tanh: `σ'(x) = 1.14393 * (1- tanh^2 ( 2/3 * x))`
/// ∴ σ'(x) = 0.6667/1.7159 * (1.7159 - σ(x)) * (1.7159 + σ(x) )
struct tanh_scaled_deriv
{
    tanh_scaled_deriv()=default;
    __device__ float operator()(const float x) const
    {
        float tanh_vl = tanhf(__fmul_rz(0.666666667,x));
        float dot = __fmul_rz((1.7159-tanh_vl),__fadd_rz(1.7159,tanh_vl));
        return __fmul_rz(0.38852303,dot);
    }
};

/// Softsign: `σ(x) = x / 1 + abs( x )`
struct soft_sign
{
    soft_sign()=default;
    __device__ float operator()(const float x) const
    {
        float denom = __fadd_rz(1,fabsf(x));
        return __fdividef(x,denom);
    }
};

/// Softsign: `σ(x) = sgn(x) / (1 + |x| )^2` where sgn is the signum
struct soft_sign_deriv
{
    soft_sign_deriv()=default;
    __device__ float operator()(const float x) const
    {
        float inner = __fadd_rz(1,fabsf(x));
        float in_sq = __fmul_rz(inner,inner);
        return __fdividef(sgn(x),in_sq);
    }
};


/** 
 * @brief Activation Kernel 
 * @param func is the activation function, same as template typename F
 * @param input is an input device array,
 * @note this is a 1D grid kernel
 */
template <typename F>
__global__ void activate(
                          F const& func, 
                          float * input
                        )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    input[x]  = func(input[x]);
}

template <typename F>
__global__ void activate(
                          F const& func, 
                          float * input, 
                          float * output
                        )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    output[x]  = func(input[x]);
}

/// TODO: Add documentation, this is ex-`sigmoid_prime`
template <typename F>
__global__ void derivatives(
                              F const& func, 
                              float * sum_ji,
                              float * output
                           )
{
    // Iterate Vector `Σ[ji]`
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    output[x] = func(sum_ji[x]);
}

/**
 * @brief Delta Error of output layer: `-E * σ'( Σ(W[i]*O[j]) )`
 * @param sum is array of previous layer output dot incoming weights: `Σ ( W[i] * O[j])`
 * @param ideal is array `ideal` output, e.g., the Target output
 * @param actual is array `actual` output
 * @param delta is array of δ[i] of output layer
 * @param index is (???) @see ann/ann.cu
 * @note all parameters (w_sum,out_err,delta) should be the same size
 */
__global__ void delta_output (
                                const float * primed_sum,
                                const float * ideal,
                                const float * actual,
                                float * delta,
                                unsigned int index
                             );


///
/// CUDA kernels for Testing a network
///

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




///
/// CUDA kernels for Training a network
///

/** 
 * @brief Calculate Delta dot product of weights and next layer node Deltas: `W[ik] * δ[k]`
 * @note We multiply each weight, with its respective (next layer node's) δ[k]
 * @note `weights per node` must equal δ[k]
 * @note Y grid is `weights per node` as well as δ[k] size (since they are equal)
 * @note X grid is layer i nodes (could differ from k nodes)
 */
__global__ void delta_product (
                                const float * w_ik,
                                const float * d_k,
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
 * @param gradient will be updated (the storage array)
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

// TODO: Resilient Back-Prop

/// Calculate the Output Error: (Ideal[i] - Actual[i])^2
__global__ void squared_error ( 
                                const float * ideal,
                                float * actual, 
                                float * errors
                              );

};
#endif
