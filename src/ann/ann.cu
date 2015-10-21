#include "ann.hpp"

/// This is the implementation cubin, which uses template classes
namespace cuANN
{

__host__ ann::ann (
                    unsigned int input_neurons,
                    unsigned int hidden_neurons,
                    unsigned int hidden_layers,
                    unsigned int output_neurons
                  )
: input_( input_neurons ), hidden_( hidden_neurons ), layers_ ( hidden_layers ), output_ ( output_neurons )  
{
    // allocate input weights
    w_input_ = thrust::device_vector<float>( input_ );
    std::cout << "input neurons: " << input_ << std::endl;

    // allocate output weights
    w_output_ = thrust::device_vector<float>( output_ ); 
    std::cout << "output neurons: " << output_ << std::endl;

    // calculate hidden weights per layer
    per_layer_ = std::ceil( hidden_ / layers_ );

    // NOTE: due to ceil, the actual number of hidden weights may change
    hidden_ = per_layer_ * layers_;

    std::cout << "hidden neurons: " << hidden_ << " per layer: " << per_layer_ << std::endl;

    if ( hidden_neurons > 0 )
        w_hidden_ = thrust::device_vector<float>( hidden_ );
    
    // low and upper random bounds
    float upper = .1f;
    float lower = -.1f;

    thrust::counting_iterator<float> index_sequence_begin(0);
    auto now = std::chrono::system_clock::now();
    auto seed = std::chrono::duration_cast<std::chrono::milliseconds>( now.time_since_epoch()).count();

    // Random Init input weights
    thrust::transform(  index_sequence_begin,
                        index_sequence_begin + w_input_.size(),
                        w_input_.begin(), 
                        prg( upper, lower, seed ) );

    // Random Init output weights
    thrust::transform(  index_sequence_begin,
                        index_sequence_begin + w_output_.size(),
                        w_output_.begin(), 
                        prg( upper, lower, seed ) );

    if ( hidden_neurons > 0 )
    {
        // Random Init all hidden weights (regardless of layer index) 
        thrust::transform(  index_sequence_begin,
                            index_sequence_begin + w_hidden_.size(),
                            w_hidden_.begin(),
                            prg( upper, lower, seed ) );
    }

    //std::cout << "input weights" << std::endl;
    //for ( int i = 0; i < w_input_.size(); i++ ) std::cout << w_input_[i] << std::endl;
    //std::cout << "hidden weights" << std::endl;
    //for ( int i = 0; i < w_hidden_.size(); i++ ) std::cout << w_hidden_[i] << std::endl;
    //std::cout << "output weights" << std::endl;
    //for ( int i = 0; i < w_output_.size(); i++ ) std::cout << w_output_[i] << std::endl;
    //std::cout << std::endl;
}


__host__ ann::h_vector ann::propagate ( ann::d_vector input ) const
{
    if ( input.size() != w_input_.size() )
        throw std::runtime_error( "ann::propagate param input size doesn't match input layer size" );

    // NOTE: If I feel the need to add a Bias Neuron, its very simple: 
    //       at any point where I have the vector `out` simply add at the end, a `1.f` value

    // put it through the input weights
    thrust::device_vector<float> out = prop_layer( w_input_, input );

    // if we have no hidden neurons
    if ( hidden_ == 0 )
        return prop_layer( w_output_, out );

    // if we do have hidden neurons
    else
    {
        // iterate hidden layers and propagate every time
        for ( int i = 0; i < layers_; i++ )
        {
            // calculate hidden layer weight start and end
            unsigned int start = i * per_layer_;
            unsigned int end = ( i * per_layer_ ) + ( per_layer_ );
            //std::cout << "start: " << start << " & end: " << end << std::endl;

            // copy as tmp hidden weight vector
            thrust::device_vector<float> hidden( w_hidden_.begin() + start, 
                                                 w_hidden_.begin() + end );
            //std::cout << "hidden layer size: " << hidden.size() << std::endl;
 
            // run propagation through that hidden layer
            // NOTE:
            //      alternatively, in ordfer to avoid copying by iterators
            //      I can change the params of ann::prop_layer, to use Iterators
            //      This would be an overloaded method, used only for hidden layers
            out = prop_layer( hidden, out );
        }

        // finally put through output layer
        return prop_layer( w_output_, out );
    }
}


//__host__ float ann::epoch (
//                                const cuANN::data & input,
//                                const float stop_error,
//                                const float alpha
//                            )
//{
    // TODO: run an epoch: update all weights, calculate MSE (batch)
    //       then see if its less than stop error
//    return 0.f;
//}


__host__ ann::d_vector ann::prop_layer ( 
                                            ann::d_vector weights,
                                            ann::d_vector input
                                       ) const
{
    // output size
    unsigned int o_size = weights.size() * input.size();

    // vectorized matrix output
    thrust::device_vector<float> mtx_output( o_size );

    // Get raw pointers for CUDA kernel
    float * i_ptr = thrust::raw_pointer_cast( input.data() );
    float * w_ptr = thrust::raw_pointer_cast( weights.data() );
    float * o_ptr = thrust::raw_pointer_cast( mtx_output.data() );

    // Calculate block theads and block number: @see cuANN::dim_find
    auto dm = dim_find_2d( weights.size(), input.size() );
    //std::cout << "thread_blocks_x: " << dm.thread_blocks_x << " & thread_blocks_y: " << dm.thread_blocks_y << std::endl;
    //std::cout << "num_blocks_x: " << dm.num_blocks_x << " & num_blocks_y: " << dm.num_blocks_y << std::endl;

    // set the threads per block and number of blocks
    dim3 threadsPerBlock( dm.thread_blocks_x, dm.thread_blocks_y );
    dim3 numBlocks( dm.num_blocks_x, dm.num_blocks_y );

    // Multiply each weight with each input, resulting in matrix mtx_output
    prop_matrix<<<numBlocks,threadsPerBlock>>>( w_ptr, i_ptr, o_ptr, weights.size(), input.size() );

    // Save the Sigmoid of the Sums in the Output
    thrust::device_vector<float> output ( weights.size() );

    // Iterate in increments of Weights ( rows )
    for ( int i = 0; i < weights.size(); i++ )
    {
        // calculate where row starts and ends
        unsigned int start = i * input.size();
        unsigned int end = ( i * input.size() ) + (input.size());
        //std::cout << "prop_layer row start: " << start << " & end: " << end << std::endl;
        //thrust::device_vector<float> row ( mtx_output.begin() + start, mtx_output.begin() + end );
        //std::cout << "matrix row size: " << row.size() << std::endl;

        // get the sum
        //float sum = thrust::reduce( row.begin(), row.end(), (float)0, thrust::plus<float>() );
        float sum = thrust::reduce( mtx_output.begin() + start, 
                                    mtx_output.begin() + end, 
                                    (float)0, 
                                    thrust::plus<float>() );

        // Sigmoid and set
        output[i] = sigmoid( sum );
    } 

    return output;
}

};
