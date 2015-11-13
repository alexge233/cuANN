#include "kernel.hpp"

namespace cuANN
{

__global__ void sigmoid_activation( 
                                    float * input, 
                                    unsigned int size 
                                  )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x < size )
    {
        float t = input[x];
        input[x]  = __fdividef( 1.f,(1.f + __expf(-1.f*t)));
        //printf("F(Σ[ji]): %f\n",input[x]);
    }
}

__global__ void sigmoid_derivative(
                                    float * sum_ji,
                                    float * output
                                  )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // f'(Σ[ji]) - Sigmoid Prime of Σ[ji]
    float t = __fdividef( 1.f, (1.f + __expf(-1.f*sum_ji[x])));
    output[x] = t * (1.f - t);
    //printf("F'(Σ[ji]): %f\n",output[x]);
}

__global__ void forward_prop ( 
                               const float * weight, 
                               const float * input, 
                               float * output, 
                               unsigned int w_size
                             )
{
    // X is input size (w_size)
    // Y is weights per neuron/node (i_size)
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    //  I[j] * W[i]
    output[w_size * x + y] = __fmul_rz(input[x], weight[w_size * x + y]);
    //printf("X: %d, Y: %d, I: %f, W: %f, O: %f\n", 
    //       x, y, input[x], weight[w_size * x + y], output[w_size * x + y] );
}

//TODO: WARNING If we obtain a value that is in the "hundreads" then the BUG is probably in here
__global__ void sum_columns ( 
                                float * w_mtx,
                                float * output, 
                                unsigned int w_size
                            )
{
    // X thread iterates the rows (the output)
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    float total;
    for ( int y = 0; y < w_size; y++ )
    {
        //printf("M: %f\n",w_mtx[ y + w_size * x]);
        total = __fadd_rz( total, w_mtx[ y + w_size * x]);
    }
    output[x] = total;
}

__global__ void delta_output (
                                float * sum,
                                float * ideal,
                                float * actual,
                                float * delta,
                                unsigned int index
                             )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    float error = -1.f * ( actual[x] - ideal[x]);
    
    // Sigmoid Prime
    float sig = __fdividef( 1.f, (1.f + __expf(-1.f*sum[x+index])));
    float primed = sig * ( 1.f - sig );
    
    // -E * F'(Actual-ideal)
    delta[x+index] = error * primed;
    //printf( "Ideal: %f, Actual: %f, Out: %f, -Error: %f, F'(Out): %f, Delta: %f\n",
    //         ideal[x], actual[x], sum[x+index],error,primed,delta[x+index]);
}

__global__ void delta_product (
                                float * w_ik,
                                float * d_k,
                                float * output,
                                unsigned int w_size
                              )
{
    // X is layer[i] nodes (size_i)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    // Y is layer[k] nodes (size_k) == d_k == w_per_n
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //  W[ik] * δ[k]
    output[w_size * x + y] = __fmul_rz( d_k[y], w_ik[w_size * x + y]);
    //printf("X: %d, Y: %d, δ[k]: %f, w[ik]: %f, dot: %f\n", 
    //        x, y, d_k[y], w_ik[w_size * x + y], output[w_size * x + y] );
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
    delta_i[x] = __fmul_rz( prime_ji[x], rhs );
    //printf("X %d, F'(Σ[ji]): %f, Σ(w[ik]*δ[k]): %f, δ[i]: %f\n",
    //        x, prime_ji[x], rhs, delta_i[x]);
}

__global__ void squared_error ( 
                                float * ideal,
                                float * actual, 
                                float * errors
                            )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    float diff = ideal[x] - actual[x];
    errors[x] = diff * diff;
}

};
