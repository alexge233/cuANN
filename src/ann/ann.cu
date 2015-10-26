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
    // If we have hidden neurons - TODO: Update formula without INPUT WEIGHTS
    if ( hidden_neurons_ > 0 )
    {
        per_layer_ = std::ceil( hidden_neurons_ / hidden_layers_ );
        hidden_neurons_ = per_layer_ * hidden_layers_;

        // Hidden weights = ( per_layer^2 * (hidden_layers -1) ) + (per_layer * output)
        //                  the first line calculates weights from input to first hiddden
        //                  the second line calculates weights within hidden layers
        //                  the third line calculates weights from hidden to output neurons
        unsigned int i_w = input_neurons_ * per_layer_;
        unsigned int h_w = std::pow( per_layer_, 2 ) * (hidden_layers_ -1 );
        unsigned int o_w = per_layer_ * output_neurons_ ;

        weights_ = thrust::device_vector<float>( i_w + h_w + o_w );

        // Index the weights appropriately
        w_index_.push_back( std::make_pair( 0, i_w ) );

        for ( int i = 1; i <= hidden_layers_; i++ )
        {
            int h = std::pow( per_layer_, 2 );
            unsigned int k = i * h;
            unsigned int left = weights_.size() - k;
            if ( left >= h )
                w_index_.push_back( std::make_pair( k, k + h ) );

            else
                w_index_.push_back( std::make_pair( k, weights_.size() ) );
        }

    }
    // no hidden neurons
    else
    {
        per_layer_ = 0;
        weights_ = thrust::device_vector<float>( input_neurons_ * output_neurons );
        w_index_.push_back( std::make_pair( 0, weights_.size() ) );
    }
   
    // low and upper random bounds
    float upper = .2f;
    float lower = -.2f;

    thrust::counting_iterator<float> index_sequence_begin(0);
    auto now = std::chrono::system_clock::now();
    auto seed = std::chrono::duration_cast<std::chrono::milliseconds>( now.time_since_epoch()).count();

    // Random Init all hidden weights (regardless of layer index) 
    thrust::transform(  index_sequence_begin,
                        index_sequence_begin + weights_.size(),
                        weights_.begin(),
                        prg( upper, lower, seed ) );

    std::cout << "input neurons: " << input_neurons << std::endl;
    std::cout << "hidden neurons: " << hidden_neurons_ << " (per layer: " << per_layer_ << ")" << std::endl;
    std::cout << "output neurons: " << output_neurons << std::endl;
    std::cout << "weights: " << weights_.size() << std::endl;

    /*
    for ( auto & pair : w_index_ )
        std::cout << "weight start: " << std::get<0>( pair ) 
                  << " weight end: " << std::get<1>( pair ) << std::endl;
     */
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

    // NOTE - COMMENTED FOR TESTING - UNCOMMENT !
    // Put the input values Through the Sigmoid Function
    //auto dim = dim_find_1D( input.size() );
    //float * input_ptr = thrust::raw_pointer_cast( input.data() );
    //sigmoid_kernel<<<dim.num_blocks_x,dim.block_threads_x>>>( input_ptr, input.size() );

    // propagate through the first (input to hidden/output)
    // if this is the only layer, this is our only propagation
    thrust::device_vector<float> out = prop_layer( std::get<0>(w_index_[0]),
                                                   std::get<1>(w_index_[0]),
                                                   input );

    // Repeat for hidden layers
    for ( int i = 1; i < w_index_.size(); i++)
    {
        out = prop_layer( std::get<0>(w_index_[i]),
                          std::get<1>(w_index_[i]),
                                            input );
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
                                          unsigned int weights_begin,
                                          unsigned int weights_end,
                                          ann::d_vector input
                                       ) const
{
    // TODO: ADD BIAS NEURON FOR EACH AND EVERY INPUT 
    //       MAKE SURE THAT ONE NEURON FIRES A VALUE OF 1.f

    // WARNING - The only reason I create a new Device Vector here
    //           Is because I don't know how to get a raw pointer from a specific range.
    // TODO: Ask on Stackoveflow.com if that is possible
    thrust::device_vector<float> weights ( weights_.begin() + weights_begin,
                                           weights_.begin() + weights_end );

    std::cout << "weights: " << std::endl;
    for ( auto val : weights )
        std::cout << val << std::endl;

    std::cout << "input: " << std::endl;
    for ( auto val : input )
        std::cout << val << std::endl;

    // Vectorized Matrix Output
    unsigned int w_per_i = weights.size() / input.size();
    thrust::device_vector<float> mtx_output( weights.size() );

    // Get raw pointers for CUDA kernel
    float * input_ptr = thrust::raw_pointer_cast( input.data() );
    float * weight_ptr = thrust::raw_pointer_cast( weights.data() );
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
    prop_matrix<<<nB1,tPb1>>>( weight_ptr, input_ptr, mtx_ptr, w_per_i, input.size() );

    std::cout << "output mtx: " << std::endl;
    int k = 0;
    for ( auto val : mtx_output )
    {
        std::cout << val << " ";
        k++;
        if ( k == w_per_i )
        {
            k = 0;
            std::cout << std::endl;
        }
    }

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
