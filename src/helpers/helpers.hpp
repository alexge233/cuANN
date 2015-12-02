#ifndef _cuANN_helpers_HPP_
#define _cuANN_helpers_HPP_
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
};
#endif
