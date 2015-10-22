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
: input_neurons_( input_neurons ), 
  hidden_neurons_( hidden_neurons ), 
  hidden_layers_ ( hidden_layers ), 
  output_neurons_ ( output_neurons )
{
    // calculate hidden weights per layer
    per_layer_ = std::ceil( hidden_neurons_ / hidden_layers_ );
    // NOTE: due to ceil, the actual number of hidden neurons  may change
    hidden_neurons_ = per_layer_ * hidden_layers_;

    // If we have hidden neurons
    if ( hidden_neurons_ > 0 )
    {
        // Hidden weights = ( per_layer^2 * (hidden_layers -1) ) + (per_layer * output)
        unsigned int h_w = std::pow( per_layer_, 2 ) * (hidden_layers_ -1 )
                         + (per_layer_ * output_neurons_ );

        weights_hidden_ = thrust::device_vector<float>( h_w );

        // Input weights = Input neurons * hidden neurons per layer
        weights_input_ = thrust::device_vector<float>( input_neurons_ * per_layer_ ); 
    }
    // no hidden neurons
    else
        // Input weights = Input neurons * output neurons
        weights_input_ = thrust::device_vector<float>( input_neurons_ * output_neurons );
   
    // low and upper random bounds
    float upper = .1f;
    float lower = -.1f;

    thrust::counting_iterator<float> index_sequence_begin(0);
    auto now = std::chrono::system_clock::now();
    auto seed = std::chrono::duration_cast<std::chrono::milliseconds>( now.time_since_epoch()).count();

    // Random Init input weights
    thrust::transform(  index_sequence_begin,
                        index_sequence_begin + weights_input_.size(),
                        weights_input_.begin(), 
                        prg( upper, lower, seed ) );

    if ( hidden_neurons > 0 )
    {
        // Random Init all hidden weights (regardless of layer index) 
        thrust::transform(  index_sequence_begin,
                            index_sequence_begin + weights_hidden_.size(),
                            weights_hidden_.begin(),
                            prg( upper, lower, seed ) );
    }

    //std::cout << "input weights" << std::endl;
    //for ( int i = 0; i < w_input_.size(); i++ ) std::cout << w_input_[i] << std::endl;
    //std::cout << "hidden weights" << std::endl;
    //for ( int i = 0; i < w_hidden_.size(); i++ ) std::cout << w_hidden_[i] << std::endl;
    std::cout << "input neurons: " << input_neurons << std::endl;
    std::cout << "hidden neurons: " << hidden_neurons_ << " (per layer: " << per_layer_ << ")" << std::endl;
    std::cout << "output neurons: " << output_neurons << std::endl;
    std::cout << "input weights: " << weights_input_.size() << std::endl;
    std::cout << "hidden weights: " << weights_hidden_.size() << std::endl;
}


__host__ ann::h_vector ann::propagate ( ann::d_vector input ) const
{
    if ( input.size() != input_neurons_ )
        throw std::runtime_error( "ann::propagate param input size doesn't match input layer size" );

    // NOTE: If I feel the need to add a Bias Neuron, its very simple: 
    //       at any point where I have the vector `out` simply add at the end, a `1.f` value

    // put it through the input weights
    thrust::device_vector<float> out = prop_layer( weights_input_, input, cuANN::sigmoid_func );

    // if we do have hidden neurons
    if ( hidden_neurons_ > 0 )
    {
        for ( int i = 0; i < hidden_layers_; i++ )
        {
            // increment by layer's full connections
            int h_w = std::pow( per_layer_, 2 );
            // find current position
            unsigned int k = i * h_w;
            // find how many weights are left
            unsigned int left = weights_hidden_.size() - k;
            // enough weights left - send the block for this layer
            if ( left >= h_w )
            {
                thrust::device_vector<float> hidden( weights_hidden_.begin() + k, 
                                                     weights_hidden_.begin() + (k + h_w) );
                out = prop_layer( hidden, out, cuANN::sigmoid_func );
            }
            // not enough weights left - send whatever weights we have left, they are the last
            // connecting to the output neurons
            else
            {
                thrust::device_vector<float> hidden( weights_hidden_.begin() + k, 
                                                     weights_hidden_.end() );
                out = prop_layer( hidden, out, cuANN::sigmoid_func );
            }
        }
    }
    return out;
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


__host__ ann::d_vector ann::prop_layer ( 
                                            ann::d_vector weights,
                                            ann::d_vector input,
                                            std::function<float(float)> activation_func
                                       ) const
{
    unsigned int w_per_i = weights.size() / input.size();
    // vectorized matrix output
    thrust::device_vector<float> mtx_output( weights.size() );

    // Get raw pointers for CUDA kernel
    float * i_ptr = thrust::raw_pointer_cast( input.data() );
    float * w_ptr = thrust::raw_pointer_cast( weights.data() );
    float * o_ptr = thrust::raw_pointer_cast( mtx_output.data() );

    // Calculate block theads and block number
    // Our X grid, is the Input size, our Y grid, is the number of weights per Input
    auto dm = dim_find_2d( input.size(), w_per_i );

    // set the threads per block and number of blocks
    dim3 threadsPerBlock( dm.thread_blocks_x, dm.thread_blocks_y );
    dim3 numBlocks( dm.num_blocks_x, dm.num_blocks_y );

    // Multiply Each Input, with its Row of Weights (Matrix of Weights) resulting in a Matrix of Sums (per Input/Row)
    prop_matrix<<<numBlocks,threadsPerBlock>>>( w_ptr, i_ptr, o_ptr, w_per_i, input.size() );

    // USED ONLY FOR TESTING THE PROPAGATION !!!
    thrust::device_vector<float> output ( input.size() );
    thrust::fill( output.begin(), output.end(), 1.f);

    // TODO: Since I now have a Matrix of Outputs, where each Row represents: Input * Weights (I[i] * W[i])
    //       I can instead use a 2D grid kernel, and Sum each Row using X to represent Row, and Y to represent Output (X,Y)
    //       and Finally, I can pass the resulting Output vector, through a Sigmoid Kernel (as a 1D Grid).
    // TODO: Write a CUDA __global__ kernel which will iterate mtx_output, by ( input.size (X) * w_per_i (Y)
    //       and Sum each ROW into an Output VECTOR.
    //       Then we sigmoid each entry in the output vector.

    /*
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

        // run the sum through the activation function
        output[i] = activation_func( sum );
    } 
    */

    return output;
}

};
