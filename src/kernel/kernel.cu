#include "kernel.hpp"

namespace cuANN
{

__device__ __constant__ float Euler = 2.71828182845904523536;

__global__ void sigmoid_activation( float * input )
{
    // Iterate Input vector
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    float _x_ = input[x];

    // -X: Neg X
    float x_neg = __fmul_rz( -1.f, _x_ );

    // Y: Euler Pow To X Negative
    float e_to_x_neg = __powf( Euler, x_neg );

    // 1 + Euler^( -X )
    float denom = __fadd_rz( 1.f, e_to_x_neg );

    // 1 / 1 + Euler^( -X )
    input[x]  = __fdividef( 1.f, denom );

//    printf("x: %.9g, -x: %.9g, E^(-x): %.9g, 1+E^(-x): %.9g, σ(x): %.9g\n",
//            _x_,x_neg,e_to_x_neg,denom,input[x]);
}

__global__ void sigmoid_prime  (
                                  float * sum_ji,
                                  float * output
                               )
{
    // Iterate Vector `Σ[ji]`
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // -X: Neg X
    float x_neg = __fmul_rz( -1.f, sum_ji[x] );

    // Y: Euler Pow To X Negative
    float e_to_x_neg = __powf( Euler, x_neg );

    // 1 + Euler^( -X )
    float denom = __fadd_rz( 1.f, e_to_x_neg );

    // 1 / 1 + Euler^( -X )
    float sig_x  = __fdividef( 1.f, denom );

    // Sigmoid Prime: σ(x) * (1 - σ(x))
    output[x] = __fmul_rz( sig_x, (1.f - sig_x) );

//    printf("x: %.9g, -x: %.9g, E^(-x): %.9g, 1+E^(-x): %.9g, σ(x): %.9g, σ'(x): %.9g\n",
//            sum_ji[x],x_neg,e_to_x_neg,denom,sig_x,output[x]);

}

__global__ void forward_prop ( 
                               const float * weight, 
                               const float * input, 
                               float * output, 
                               unsigned int w_size
                             )
{
    // X is input size (w_size)
    int x = blockIdx.x * blockDim.x + threadIdx.x; 

    // Y is weights per neuron/node (i_size)
    int y = blockIdx.y * blockDim.y + threadIdx.y;   

    //  I[j] * W[i] - Row-Major Matrix
    output[w_size*x+y] = __fmul_rz(input[x], weight[w_size * x + y]);
    
//    printf("X: %d, Y: %d, I: %.9g, W: %.9g, O: %.9g\n", 
//            x, y, input[x], weight[w_size * x + y], output[w_size * x + y] );
}

//TODO: BUG: Sum Coumns NOT ROWS!
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
//        printf("X: %d, O[j]*W[i]: %.9g\n",x,w_mtx[y*width+x]);
        total = __fadd_rz( total, w_mtx[y*width+x]);
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
    // x is the output neuron/node count (e.g., length of actual & ideal)
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the Negative Error: -(Actual - Ideal)
    float neg_error = __fmul_rz( -1.f, ( actual[x] - ideal[x]) );

    // -X: Neg X
    float x_neg = __fmul_rz( -1.f, sum[x+index] );

    // Y: Euler Pow To X Negative
    float e_to_x_neg = __powf( Euler, x_neg );

    // 1 + Euler^( -X )
    float denom = __fadd_rz( 1.f, e_to_x_neg );

    // 1 / 1 + Euler^( -X )
    float sig_x  = __fdividef( 1.f, denom );

    // Sigmoid Prime: σ(x) * (1 - σ(x))
    float primed = __fmul_rz( sig_x, (1.f - sig_x) );

    // -E * σ'(Σ(O[i])
    delta[x+index] = __fmul_rz( neg_error, primed );

//    printf( "Ideal: %.9g, Actual: %.9g, Out: %.9g, -Error: %.9g, F'(Out): %.9g, Delta: %.9g\n",
//             ideal[x], actual[x], sum[x+index],neg_error,primed,delta[x+index]);
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
    
//    printf("X:%d,Y:%d, δ[k]: %.9g, w[ik]: %.9g, dot: %.9g\n", 
//            x, y, d_k[y], w_ik[width*x+y], output[width*x+y]);
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
//    printf("X:%d, δ[i]: %.9g\n",x,delta_i[x]);
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

//    printf("X %d, F'(Σ[ji]): %.9g, Σ(w[ik]*δ[k]): %.9g, δ[i]: %.9g\n",
//            x, prime_ji[x], rhs, delta_i[x]);
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

//    printf("X: %d, Y: %d, δ[k]: %.9g, O[i]: %.9g, dot: %.9g\n", 
//            x, y, d_k[x], o_i[y], (d_k[x]*o_i[y]) );
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
