#ifndef _cuANN_sigmoid_HPP_
#define _cuANN_sigmoid_HPP_
#include "includes.ihh"
namespace cuANN
{

/// Sigmoid Activation Function
struct sigmoid
{
    sigmoid () = default;

     __host__ __device__ __forceinline__ float operator()( const float x ) const
    {
        return 1.0 / (1.0 + exp ( -x ) );
    }

};

/// Fast Sigmoid Activation Function
struct fast_sigmoid
{
    fast_sigmoid() = default;

    __host__ __device__ __forceinline__ float operator()( const float x ) const
    {
        return x / ( 1 + abs( x ) );
    }

};

}
#endif
