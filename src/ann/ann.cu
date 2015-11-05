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
    if ( hidden_neurons == 0 && hidden_layers != 0 )
        throw std::runtime_error ( "you can't have no hidden neurons without hidden layers" );

    if ( hidden_neurons != 0 && hidden_layers == 0 )
        throw std::runtime_error ( "you can't have hidden neurons without hidden layers" );

    // If we have hidden neurons
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

        // alloc weights
        weights_ = thrust::device_vector<float>( i_w + h_w + o_w );

        // Index the weights appropriately
        w_index_.push_back( std::make_pair( 0, i_w ) );

        // weight count (update to keep track of previous weight index)
        unsigned int w_count = i_w;

        for ( int i = 1; i <= hidden_layers_; i++ )
        {
            // This layer's hidden neurons
            int w_layer = (per_layer_ * per_layer_);
            // index: weights_[from]
            unsigned int from = w_count;
            // weights still needed
            unsigned int left = weights_.size() - from;
            if ( left > w_layer )
            {
                w_index_.push_back( std::make_pair( from, from + w_layer ) );
                w_count += w_layer; 
            }
            else
            {
                w_index_.push_back( std::make_pair( from, weights_.size() ) );
                w_count += w_layer;
            }
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
    float upper = .1f;
    float lower = -.1f;

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
}


__host__ float ann::train (
                              const cuANN::data & train_data,
                              float mse_stop,
                              unsigned int epochs
                          )
{
    // Load the training input data in device memory
    thrust::device_vector<float> input_rows( train_data.size() * train_data.input_size() );
    // Load the training output data in host memory
    thrust::host_vector<float> output_rows( train_data.size() * train_data.output_size() );

    for ( int i = 0; i < train_data.size(); i++ )
    {
        thrust::copy( train_data[i].input.begin(), 
                      train_data[i].input.end(),
                      input_rows.begin() + (i * train_data.input_size()) );
        thrust::copy( train_data[i].output.begin(),
                      train_data[i].output.end(),
                      output_rows.begin() + (i * train_data.output_size()) );
    }

    if ( input_rows.size() == output_rows.size() )
        throw std::runtime_error("input rows not equal to output rows");

    float mse = 0.f;

    // Iterate epochs and compare mse
    for ( int i = 0; i < epochs; i++ )
    {
        mse = epoch( input_rows,
                     train_data.input_size(),
                     output_rows,
                     train_data.output_size(),
                     train_data.size() );

        // NOTE/TODO: early stopping requires a validation set, which is different from the training set
        //            we can use it to calculate the output error (MSE) from validation test instead of the training set
        if ( mse < mse_stop )
            break;
    }

    return mse;
}

__host__ ann::h_vector ann::propagate ( ann::d_vector input ) const
{
    if ( input.size() != input_neurons_ )
        throw std::runtime_error( "ann::propagate param input size doesn't match input layer size" );

    // Put the input through the activation function
    thrust::device_vector<float> output = input;
    float * output_ptr = thrust::raw_pointer_cast( output.data() );
    auto dim = dim_find_1D( output.size() );
    sigmoid_kernel<<<dim.num_blocks_x,dim.block_threads_x>>>( output_ptr, output.size() );

    // Propagate Input through every layer.
    // If we have no hidden, then propagate through the weights from input to output layers
    for ( int i = 0; i < w_index_.size(); i++)
    {
        // NOTE: TODO - Consider putting all code into a single kernel (0.2 release)
        //              Consider putting the loop and all code into a single kernel

        unsigned int w_from = std::get<0>(w_index_[i]);
        unsigned int w_to = std::get<1>(w_index_[i]);
    
        // Update Out to equal the propagation of previous out.
        auto sums = prop_layer( w_from,
                                w_to,
                                output );

        // Copy the sums as output
        output = sums;
    
        // Run the output through the activation function
        auto dim_B = dim_find_1D( output.size() );
        output_ptr = thrust::raw_pointer_cast( output.data() );
        sigmoid_kernel<<<dim_B.num_blocks_x,dim_B.block_threads_x>>>( output_ptr, output.size() );
    }

    return output;
}


__host__ float ann::epoch ( 
                                d_vector & input,
                                unsigned int input_len,
                                h_vector & output,
                                unsigned int output_len,
                                unsigned int total
                          )
{
    thrust::device_vector<float> errors( total );
    unsigned int delta_size = hidden_neurons_ + output_neurons_;
    thrust::device_vector<float> layer_sums(delta_size);
    thrust::device_vector<float> delta_vals(delta_size);

    // Iterate all training data once
    for ( int i = 0; i < total; i++ )
    {
        // Assign our actual output to input - We will modify this
        thrust::device_vector<float> actual_output( input.begin() + (input_len*i), 
                                                    input.begin() + (input_len*i) + input_len );

        // Activation Function for Input Values
        float * output_ptr = thrust::raw_pointer_cast( actual_output.data() );
        auto dimA = dim_find_1D( actual_output.size() );
        sigmoid_kernel<<<dimA.num_blocks_x,dimA.block_threads_x>>>( output_ptr, actual_output.size() );
         
        // propagate through every layer (Input to Hidde, Hidden to Output)
        unsigned int sums_idx = 0;
        for ( int k = 0; k < w_index_.size(); k++)
        {
            auto w_from = std::get<0>(w_index_[k]);
            auto w_to = std::get<1>(w_index_[k]);

            // Sums ( Input * Weights ) for Layer[k,j]
            auto sums = prop_layer( w_from, w_to, actual_output );

            thrust::copy( sums.begin(), sums.end(), layer_sums.begin() + sums_idx );
            sums_idx += sums.size();
            actual_output = sums;
            output_ptr = thrust::raw_pointer_cast( actual_output.data() );

            // Activation Function for Output SUMS
            auto dimB = dim_find_1D( actual_output.size() );
            sigmoid_kernel<<<dimB.num_blocks_x,dimB.block_threads_x>>>( output_ptr, actual_output.size() ); 
        }

        thrust::device_vector<float> ideal( output.begin() + (output_len*i),
                                            output.begin() + (output_len*i) + output_len );

        unsigned int from = layer_sums.size() - output_neurons_;
        float * delta_ptr = thrust::raw_pointer_cast( delta_vals.data() );
        float * sums_ptr = thrust::raw_pointer_cast( layer_sums.data() );
        float * ideal_ptr = thrust::raw_pointer_cast( ideal.data() );
        float * actual_ptr = thrust::raw_pointer_cast( actual_output.data() );

        // Calculate the Delta rule for last (Output) Layer
        auto dimC = dim_find_1D( output_neurons_ );
        delta_output<<<dimC.num_blocks_x,dimC.block_threads_x>>>(sums_ptr,ideal_ptr,actual_ptr,delta_ptr,from );

        // Calculate the Delta rule for all hidden layers
        for ( int k = (w_index_.size()-2); k >= 0; k-- )
        {
            auto w_from = std::get<0>(w_index_[k]);
            auto w_to = std::get<1>(w_index_[k]);
            // NOTE: output delta calculation requires as many threads as there are output neurons
            // TODO: run the delta_hidden_kernel - TODO: implement the delta_hidden_kernel
        }

        // If doing online training, we need to update the weights now (after calculating the gradient)
    }

    // if we calculate the gradient for the Entire Epoch it is called BATCH learning (do it here)

    // TODO: Calculate for each Gradient for each weight
    // TODO: Adjust Each weight, using the gradient

    //return sum_errors  / num_errors;
    return 0.f;
}


__host__ ann::d_vector ann::prop_layer ( 
                                          unsigned int weights_begin,
                                          unsigned int weights_end,
                                          const ann::d_vector & input
                                       ) const
{
    // TODO: ADD BIAS NEURON FOR EACH AND EVERY INPUT 
    //       MAKE SURE THAT ONE NEURON FIRES A VALUE OF 1.f
    //       Only way to do this, is to re-alloc (alloc) a new input vector here.
    unsigned int weights_size = weights_end - weights_begin;
    unsigned int w_per_i = weights_size / input.size();

    // Get raw pointers for CUDA kernel
    const float * input_ptr = thrust::raw_pointer_cast( input.data() );
    const float * weight_ptr = thrust::raw_pointer_cast( weights_.data() ) + weights_begin;

    // Allocate Vectorized Matrix and get Raw pointer
    thrust::device_vector<float> mtx_output( weights_size );
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

    // TODO: NOTE: consider putting all code (prop_matrix, sum_columns) into one kernel;
    // Sums Row vector
    thrust::device_vector<float> sums ( w_per_i );
    float * sums_ptr = thrust::raw_pointer_cast( sums.data() );
    auto dm_b = dim_find_1D( w_per_i );

    // Sum each column into a vector row : I[i] * W[i] = I[k]
    // Sumarize Columns, using Matrix as Input, Sums vector as output, 
    // where Width = Weights per I (# of Columns), and Height = # of Inputs (# of Rows )
    sum_columns<<<dm_b.num_blocks_x,dm_b.block_threads_x>>>( mtx_ptr, sums_ptr, w_per_i, input.size() );

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


};
