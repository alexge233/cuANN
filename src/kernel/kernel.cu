#include "kernel.hpp"
namespace cuANN
{
/*
/// Signum function: returns ±1 or 0
__device__ float signum(const float x)
{
    return (0.f < x) - (x < 0.f);
}

/// Hyperbolic Tangent: `tanh(x) = e^(x) - e^(-x) / e^(x) + e^(-x)i`
__device__ float tanh(const float x)
{
    float exp2x = __expf(2.f*x);
    return __fdividef((exp2x-1.f),__fadd_rz(exp2x,1.f));
}

/// Sigmoid: `σ(x) = 1 / 1 + e^( -x )`
__device__ float sigmoid(const float x)
{
    float exp_val = __expf(-x);
    float denom = __fadd_rz(1.f,exp_val);
    return __fdividef(1.f,denom);
}

/// Sigmoid Bipolar: `σ(x) = -1 + 2 / (1 + e^-x)`
__device__ float sigmoid_bipolar(const float x)
{
    float nom = __fadd_rz(-1.f,2.f);
    float denom = __fadd_rz(1.f,__expf(-x));
    return __fdividef(nom,denom);
}

/// Hyperbolic Tangent: `tanh(x) = e^(x) - e^(-x) / e^(x) + e^(-x)i`
/// Tanh Scaled: `1.7159 * tanh(2.f/3.f*x)`
/// @note: The function is scaled in range {-1,1} to avoid learning saturation
__device__ float tanh_scaled(const float x)
{
    return 1.7159f * logistic::tanh(2.f/3.f*x);
}

/// Softsign: `σ(x) = x / 1 + abs( x )`
__device__ float soft_sign(const float x)
{
    float denom = __fadd_rz(1.f,fabsf(x));
    return __fdividef(x,denom);
}

/// Sigmoid: `σ'(x) = σ(x) * (1 - σ(x) )`
__device__ float sigmoid_deriv(const float x)
{
    float sigma = sigmoid(x);
    float denom = 1.f - sigma;
    return __fmul_rz(sigma,denom);
}

/// Sigmoid Bipolar: `σ(x) = 0.5 * (1 + σ(x)) * (1 – σ(x) )`
__device__ float sigmoid_bipolar_deriv(const float x)
{
    float sigma = sigmoid(x);
    float lhs = __fmul_rz(0.5f,__fadd_rz(1.f,sigma));
    float rhs = 1.f - sigma;
    return __fmul_rz(lhs,rhs);
}

/// Tanh: `σ'(x) = 1.14393 * (1- tanh^2 ( 2/3 * x))`
__device__ float tanh_scaled_deriv(const float x)
{
    float scaled_x = __fmul_rz(__fdividef( 2.f, 3.f),x);
    float tanh_val = logistic::tanh(scaled_x);
    float tanh_sq = __fmul_rz(tanh_val,tanh_val);
    float rhs = 1.f - tanh_sq;
    return __fmul_rz(1.14393f,rhs);
}

/// Softsign: `σ(x) = sgn(x) / (1 + |x| )^2` where sgn is the signum
__device__ float soft_sign_deriv(const float x)
{
    float rhs = __fadd_rz(1.f,fabsf(x));
    float rhs_sq = __fmul_rz(rhs,rhs);
    return __fdividef(signum(x),rhs_sq);
}
*/

/*
__global__ void sigmoid_activation( float * input )
{
    // Iterate Input vector
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    // 1 / 1 + Euler^( -X )
    input[x]  = logistic::sigmoid(input[x]); 
}
*/

__global__ void forward_prop ( 
                               const float * weight, // W[ji] 
                               const float * input,  // O[j]
                               float * output,       // I[i]
                               unsigned int w_size   // weights per node (# of columns)
                             )
{
    // X is input size (w_size)
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    // Y is weights per neuron/node (i_size)
    int y = blockIdx.y * blockDim.y + threadIdx.y;   
    //  I[j] * W[i] - Row-Major Matrix
    output[w_size*x+y] = __fmul_rz(input[x], weight[w_size * x + y]);
}

__global__ void sum_columns ( 
                                float * w_mtx,
                                float * output, 
                                unsigned int height,
                                unsigned int width
                            )
{
    // X thread iterates Columns and sums their Row values
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    float total;
    for ( int y = 0; y < height; y++ )
    {
        total = __fadd_rz( total, w_mtx[y*width+x]);
    }
    output[x] = total;
}

__global__ void delta_product (
                                float * w_ik,
                                float * d_k,
                                float * output,
                                unsigned int width
                              )
{
    // X is layer[i] nodes (size_i)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    // Y is layer[k] nodes (size_k) == d_k == w_per_n
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    //  W[ik] * δ[k] - Row-Major Matrix
    output[width*x+y] = __fmul_rz( d_k[y], w_ik[width*x+y]);
}

__global__ void delta_sum_rows (
                                float * w_ik_d,
                                float * delta_i,
                                unsigned int width
                               )
{
    // X thread iterates Rows and Sums the respective Column values
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    float total = 0.f;
    for ( int y = 0; y < width; y++ )
    {
        //printf("X:%d, Σ: %.9f + %.9f\n",x,total,w_ik_d[x*width+y]);
        total = __fadd_rz( total, w_ik_d[x*width+y]);
    }
    delta_i[x] = total;
}

__global__ void delta_hidden (
                               float * prime_ji,
                               float * delta_i
                             )
{
    // X grid is size_i
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    // δ[i] = f'( Σ[ji]) * Σ(w[ik] * δ[k])
    // NOTE: delta_i ALREADY contains `Σ(w[ik] * δ[k])`
    float rhs = delta_i[x];
    // δ[i] = σ'( Σ[ji]) * Σ(w[ik] * δ[k])
    delta_i[x] = __fmul_rz( prime_ji[x], rhs );
}

__global__ void gradient_descent (
                                    float * d_k,
                                    float * o_i,
                                    float * g_ik,
                                    unsigned int size_d
                                 )
{
    // X = Node Delta Count (layer k)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    // Y = Node Output Count (layer i)
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Row-Major Matrix
    g_ik[size_d*x+y] = __fmul_rz( d_k[x], o_i[y]);
}

__global__ void sum_gradients (
                                float * gradient,
                                float * new_value
                              ) 
{
    // X Grid iterates all gradient values 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    // A Simple summation
    gradient[x] = __fadd_rz( gradient[x], new_value[x] );
}

__global__ void back_prop (
                            float * weight,
                            float * gradient,
                            float * update,
                            float alpha,
                            float epsilon
                         )
{
    // X Grid iterates weight, gradient and update (all same size)    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    // ε * ( ∂E / ∂W[ik] )
    float lhs = __fmul_rz( epsilon, gradient[x] ); 
    // α * ( Δw(t-1) )
    float rhs = __fmul_rz( alpha, update[x] );
    // Δw(t) = ε * ( ∂E / ∂W[i] ) + α * ( Δw(t-1) )
    float d_w = __fadd_rz( lhs, rhs );
    // Update weight: W[i] = W[i] + Δw(t)
    weight[x] = __fadd_rz( weight[x], d_w );
    // Set `Δw(t-1) = Δw(t)`
    update[x] = d_w;
}

__global__ void squared_error ( 
                                float * ideal,
                                float * actual, 
                                float * errors
                            )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = ideal[x] - actual[x];
    errors[x] = __fmul_rz( diff, diff );
}

};
