#ifndef _cuANN_prg_HPP_
#define _cuANN_prg_HPP_
#include "includes.ihh"
namespace cuANN
{

/// Pseudo-Random Number Generator
struct prg
{
    float a, b;

    __host__ __device__ prg( float _a=0.f, float _b=1.f ) 
    : a(_a), b(_b) 
    {};

    __host__ __device__ float operator()( const unsigned int n ) const
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<float> dist(a, b);
        rng.discard(n);
        return dist(rng);
    }
};
}

#endif
