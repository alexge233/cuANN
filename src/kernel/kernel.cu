#include "kernel.hpp"

namespace cuANN
{


__global__ void sigmoid_kernel( float * input, unsigned int size )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x < size )
    {
        float t = input[x];
        input[x]  = __fdividef( 1.f,(1.f + __expf(-1.f*t)));
        //input[x] = 1.f / (1.f + exp ( -1.f * value ) );
    }
}

__global__ void prop_kernel ( 
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
    //printf("I: %f, W: %f, O: %f\n",input[x],weight[w_size * x + y],output[w_size * x + y]);
}

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
    float t = __fdividef( 1.f, (1.f + __expf(-1.f*sum[x+index])));
    float f = t * ( 1.f - t );
    
    // -E * F'(Actual-ideal)
    delta[x+index] = error * f;
}


// WARNING - DANGER : I guarantee you that the below lines DO NOT DO what I think they do, and there is a BUG!
// TODO: Break Up in 4 smaller functions: 
//          1. F'(S[ji]) 
//          2. W[ik]*δ[k]               - Use prop_kernel if possible
//          3. Σ(W[ik]*δ[k])            - Use sum_columns if possible
//          4. F'(S[ji]) * Σ(W[ik]*δ[k])
__global__ void delta_hidden (
                               float * sum_ji,
                               float * delta_i,
                               float * delta_k,
                               float * weight_ik,
                               unsigned int size_i,
                               unsigned int size_w
                             )
{
    // X grid is layer [i] node iterator (e.g, size_i)
    // Y grid is layer [k] node iterator (e.g, size_k)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Use dynamic shared memory
    extern __shared__ float tmp[];

    // #1 f'(S[ji]) - Sigmoid Prime of S[ij] store in delta_i (our output)
    float t = __fdividef( 1.f, (1.f + __expf(-1.f*sum_ji[x])));
    delta_i[x] = t * (1.f - t);    

    // #2 store in tmp: Σ(w[ik] * δ[k]) - store in tmp: accumulate for each row (size_i) the multiplied elements (Sum)
    for ( int z = 0; z < size_w; z++ )
    {
        tmp[x] += weight_ik[z+size_w*x] * delta_k[y];
    }

    // #3 f'(S[i]) * Σ(w[ik] * δ[k])
    delta_i[x] *= tmp[x];
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
