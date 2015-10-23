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
    float upper = .2f;
    float lower = -.2f;

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


__host__ float ann::train (
                              const cuANN::data & input,
                              float mse_stop,
                              unsigned int epochs,
                              float learning,
                              float momentum
                          )
{
    // Run epoch iterations, doing a back_propagation
    // after each iteration
    // Stop only if mse_stop is achieved, or if we run out of epochs
    // TODO...
    return -1.f;
}

__host__ ann::h_vector ann::propagate ( ann::d_vector input ) const
{
    if ( input.size() != input_neurons_ )
        throw std::runtime_error( "ann::propagate param input size doesn't match input layer size" );

    // NOTE: If I feel the need to add a Bias Neuron, its very simple: 
    //       at any point where I have the vector `out` simply add at the end, a `1.f` value

    // put it through the input weights
    thrust::device_vector<float> out = prop_layer( weights_input_, input );

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
            out = prop_layer( hidden, out );
        }
        // not enough weights left - send whatever weights we have left, they are the last
        // connecting to the output neurons
        else
        {
            thrust::device_vector<float> hidden( weights_hidden_.begin() + k, 
                                                 weights_hidden_.end() );
            out = prop_layer( hidden, out );
        }
    }

    return out;
}


__host__ float ann::epoch ( const cuANN::data & input )
{
    // Accumulate squared errors
    thrust::host_vector<float> errors( input.size() );

    //  Iterate input - 
    //  NOTE: Iterating host_vectors from training data requires a lot of copying here
    //  I think its best if the input param, was a vector of pairs with device_vectors
    //  Since its simply stupid loading them from host to device at every iteration
    for ( auto & row : input )
    {
        // Propagate input - get output (WARNING: copying from host to device)
        thrust::device_vector<float> output = propagate( row.input );

        // get squared errors - (WARNING: copying from host to device)
        thrust::device_vector<float> sq_errors = output_errors( row.output, output );

        // sum up square errors for this input - push it back
        float sum_errors = thrust::reduce( sq_errors.begin(), sq_errors.end() );
        errors.push_back( sum_errors );
    }
    // sum all errors
    float num_errors = errors.size();
    float sum_errors = thrust::reduce( errors.begin(), errors.end() );

    return sum_errors  / num_errors;
}


__host__ ann::d_vector ann::prop_layer ( 
                                            ann::d_vector weights,
                                            ann::d_vector input
                                       ) const
{
    unsigned int w_per_i = weights.size() / input.size();
    // vectorized matrix output
    thrust::device_vector<float> mtx_output( weights.size() );

    // Get raw pointers for CUDA kernel
    float * inputs_ptr = thrust::raw_pointer_cast( input.data() );
    float * weights_ptr = thrust::raw_pointer_cast( weights.data() );
    float * mtx_ptr = thrust::raw_pointer_cast( mtx_output.data() );

    // Calculate block theads and block number
    // Our X grid, is the Input size, our Y grid, is the number of weights per Input
    auto dm_a = dim_find_prop_mtx( input.size(), w_per_i );

    // set the threads per block and number of blocks
    dim3 tPb1( dm_a.block_threads_x, dm_a.block_threads_y );
    dim3 nB1( dm_a.num_blocks_x, dm_a.num_blocks_y );

    // Multiply Each Input, with its Row of Weights (Matrix of Weights) resulting in a Matrix of Sums (per Input/Row)
    // This is a 2D Grid iterations, using as input Weights and Inputs, and the Matrix as output
    // Width (Columns) is # weights per input, Height (Rows) is # inputs
    prop_matrix<<<nB1,tPb1>>>( weights_ptr, inputs_ptr, mtx_ptr, w_per_i, input.size() );

    // Sums Row vector
    thrust::device_vector<float> sums ( w_per_i );
    float * sums_ptr = thrust::raw_pointer_cast( sums.data() );
    auto dm_b = dim_find_1D( w_per_i );

    // Sum each column into a vector row : I[i] * W[i] = I[k]
    // Sumarize Columns, using Matrix as Input, Sums vector as output, 
    // where Width = Weights per I (# of Columns), and Height = # of Inputs (# of Rows )
    sum_columns<<<dm_b.num_blocks_x,dm_b.block_threads_x>>>( mtx_ptr, sums_ptr, w_per_i, input.size() );

    // Finally, run the sums through the activation function
    // Same num of blocks, same number of threads per block
    sigmoid_kernel<<<dm_b.num_blocks_x,dm_b.block_threads_x>>>( sums_ptr, sums.size() ); 
   
    return sums;
}

__host__ ann::d_vector ann::output_errors (
                                              d_vector ideal,
                                              d_vector actual
                                          ) const
{
    if ( ideal.size() != actual.size() )
        throw std::runtime_error ( "cuANN::ann::output_error: ideal vector diff size from actual vector" );

    // NOTE: Running this on GPU using device vectors is probably an overkill.
    //       This should be profiled, and if found to offer no advantage, moved to host code only
    
    // This is PART of the MSE: (Ideal[i] - Actual[i])^2 + ...
    thrust::device_vector<float> errors( ideal.size() );
    float * ideal_ptr = thrust::raw_pointer_cast( ideal.data() );
    float * actual_ptr = thrust::raw_pointer_cast( actual.data() );
    float * errors_ptr = thrust::raw_pointer_cast( errors.data() );

    // First calculate all errors 
    auto dm = dim_find_1D( ideal.size() );
    squared_error<<<dm.num_blocks_x,dm.block_threads_x>>>( ideal_ptr, actual_ptr, errors_ptr );

    // This is how u sum all errors using thrust::reduce
    //float squared_error = thrust::reduce( errors.begin(), errors.end() );
    return errors;
}

ann::d_vector ann::gradient_descent (
                                       // ???
                                    )
{
    // TODO: see Heaton's video
    //       Calculate all Gradients for each Weight
}

void ann::back_prop_batch (
                          // ???
                         )
{
    // TODO:
    // Calculate for every weight it's gradient descent
    // Then update the weight using the Back-Prop algorithm
    // NOTE: See notes and Heaton's videos for more info
    // NOTE: This is the BATCH version: summ gradients before updating weights
}

void ann::back_prop_online (
                            // ???
                           )
{
    // TODO: see above^^
}

};
