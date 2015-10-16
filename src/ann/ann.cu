#include "ann.hpp"

/// This is the implementation cubin, which uses template classes
namespace cuANN
{

__host__ ann::ann ( )
{
    // set input layer neurons/weights
    input__ = thrust::device_vector<float>( input_neurons__ );

    // one hidden layer with 2 neurons/weights
    hidden__ = thrust::device_vector<float>( input_neurons__ ); 

    // Setup output layer neurons/weights
    output__ = thrust::device_vector<float>( output_neurons__ ); 
    
    // low and upper random bounds
    float upper = 1.f;
    float lower = 0.f;

    thrust::counting_iterator<unsigned int> index_sequence_begin(0);

    thrust::transform(  index_sequence_begin, 
                        index_sequence_begin + input__.size(), 
                        input__.begin(), 
                        cuANN::prg( upper, lower ) );

    thrust::transform(  index_sequence_begin,
                        index_sequence_begin + hidden__.size(), 
                        hidden__.begin(), 
                        cuANN::prg( upper, lower ) );

    thrust::transform(  index_sequence_begin,
                        index_sequence_begin + output__.size(), 
                        output__.begin(), 
                        cuANN::prg( upper, lower ) );

    std::cout << "random weights initialised" << std::endl;
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

__host__ thrust::host_vector<float> prop ( thrust::host_vector<float> test_input )
{
    // TODO: propagate values, by activating each neuron/layer and calculate output
    //       then return that output back
}

__device__ __host__ __forceinline__ float ann::sigmoid__ ( const float x ) const
{
    return 1.0 / (1.0 + exp ( -x ) );
}

__device__ __host__ __forceinline__ float ann::fast_sigmoid__ ( const float x ) const
{
    return x / ( 1 + abs( x ) );
}

};
