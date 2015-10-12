#include "ann.hpp"

/// This is the implementation cubin, which uses template classes
namespace cuANN
{

__host__ ann::ann ( 
                        const unsigned int input_neurons,
                        const unsigned int hidden_neurons,
                        const unsigned int hidden_layers,
                        const unsigned int output_neurons
                    )
{
    // TODO
}


__host__ float ann::epoch (
                                const cuANN::data & input,
                                const float stop_error,
                                const float alpha
                            )
{
    // TODO: run an epoch: update all weights, calculate MSE (batch)
    //       then see if its less than stop error
    return 0.f;
}

// CPU only
__host__ thrust::host_vector<float> test ( thrust::host_vector<float> test_input )
{
    // TODO: propagate values, by activating each neuron/layer and calculate output
    //       then return that output back
}

// GPU
__device__ __host__ __forceinline__ float ann::sigmoid__ ( const float x ) const
{
    return 1.0 / (1.0 + exp ( -x ) );
}

// GPU
__device__ __host__ __forceinline__ float ann::fast_sigmoid__ ( const float x ) const
{
    return x / ( 1 + abs( x ) );
}

// GPU
__device__ __host__ float ann::prng__ (
                                         const float min,
                                         const float max
                                      ) const
{
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist( min, max );
    //rng.discard(n);
    return dist(rng);
}

};
