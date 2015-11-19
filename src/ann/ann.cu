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

        // i = 1, because we have already set our input neurons
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
    // momentum
    alpha_ = 0.3;
    // learning rate
    epsilon_ = 0.7;

    std::cout << "input neurons: " << input_neurons << std::endl;
    std::cout << "hidden neurons: " << hidden_neurons_ << " (per layer: " << per_layer_ << ")" << std::endl;
    std::cout << "output neurons: " << output_neurons << std::endl;
    std::cout << "weights: " << weights_.size() << std::endl;
    std::cout << "learning rate: " << epsilon_ << " and momentum: " << alpha_ << std::endl;

//    for ( int k = 0; k < w_index_.size(); k++)
//    {
//        auto from = std::get<0>(w_index_[k]);
//        auto to   = std::get<1>(w_index_[k]);
//        for ( int x = from; x < to; x++ )
//            std::cout << weights_[x] << " ";
//        std::cout << std::endl;
//    }
}

__host__ float ann::train (
                              const cuANN::data & train_data,
                              float mse_stop,
                              unsigned int epochs,
                              unsigned int reports,
                              bool online
                          )
{
    // Load the training input data
    thrust::host_vector<float> input_rows( train_data.size() * train_data.input_size() );

    // Load the training output data
    thrust::host_vector<float> output_rows( train_data.size() * train_data.output_size() );

    // Make a copy of all data (because our train_data scheme uses tuples)
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

    float mse = 0.0;

    // Initialise and Resize the vector `updates_` same size as weights
    updates_ = thrust::device_vector<float>( weights_.size() );

    // Iterate epochs and compare mse
    int k = 0;
    for ( int i = 0; i < epochs; i++ )
    {
        mse = epoch( input_rows,
                     train_data.input_size(),
                     output_rows,
                     train_data.output_size(),
                     train_data.size(),
                     online );
        if ( k == reports && k != 0 )
        {
            std::cout << "Epoch "<< i << " MSE: " << mse  << std::endl;
            k = 0;
        }
        k++;
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

    auto dimA = dim1D(output.size());
    sigmoid_activation<<<dimA.num_blocks_x,dimA.block_threads_x>>>( output_ptr );

    // Propagate Input through every layer.
    // If we have no hidden, then propagate through the weights from input to output layers
    for ( int i = 0; i < w_index_.size(); i++)
    {
        unsigned int w_from = std::get<0>(w_index_[i]);
        unsigned int w_to = std::get<1>(w_index_[i]);

        // Update Out to equal the propagation of previous out.
        auto sums = prop_layer( w_from, w_to, output );
        
        // Copy the sums as output
        output = sums;

        // Run the output through the activation function
        auto dimB = dim1D(output.size());
        sigmoid_activation<<<dimB.num_blocks_x,dimB.block_threads_x>>>( output_ptr );
    }

    return output;
}

__host__ float ann::epoch ( 
                                h_vector & input,
                                unsigned int input_len,
                                h_vector & output,
                                unsigned int output_len,
                                unsigned int total,
                                bool online
                          )
{
    // Calculate Delta Nodes size
    unsigned int delta_size = hidden_neurons_+output_neurons_;
    // Layer Sums: Σ( O[j] * W[ij] )
    thrust::device_vector<float> layer_sums(delta_size);
    float * sums_ptr = thrust::raw_pointer_cast( layer_sums.data() );
    // Delta (t) Values: δ[i] & δ[k]
    thrust::device_vector<float> delta_vals(delta_size);
    float * delta_ptr = thrust::raw_pointer_cast( delta_vals.data() );
    // Primed Sums: F'( Σ( O[j] * W[ij] ) )
    thrust::device_vector<float> primed_sums(delta_size);
    float * primed_ptr = thrust::raw_pointer_cast( primed_sums.data() );
    // Ideal (Target) Output
    thrust::device_vector<float> ideal(output_neurons_);
    float * ideal_ptr = thrust::raw_pointer_cast( ideal.data() );
    // Store the Output (O[i]) for each Neuron/Node
    thrust::device_vector<float> layer_outputs(input_neurons_+hidden_neurons_+output_neurons_);
    // Batch Summed Gradients (Row-Major Matrix)
    thrust::device_vector<float> gradients;
    if ( !online ) gradients = thrust::device_vector<float>( weights_.size() );
    // All Pattern Errors for all output nodes
    thrust::device_vector<float> errors(total*output_neurons_);
    // Wait for all allocations
    cudaDeviceSynchronize();

    // AT THIS POINT, running Abelone seems very slow.
    // Only 35% of GPU (GTX660) is occupied, and only 25% of CPU.
    // I probably can Multi-thread the Loop Below, and Run Multiple Pattern Training Threads on the GPU.

    // Iterate all training data once 
    for ( int i = 0; i < total; i++ )
    {
        // Copy input from Host to Device as Output - We will modify contents & size
        thrust::device_vector<float> actual_output( input.begin() + (input_len*i), 
                                                    input.begin() + (input_len*i) + input_len );
        float * output_ptr = thrust::raw_pointer_cast( actual_output.data() );
        cudaDeviceSynchronize();
        // Activation Function for Input Values
        auto dimA = dim1D(actual_output.size());
        sigmoid_activation<<<dimA.num_blocks_x,dimA.block_threads_x>>>( output_ptr );
        cudaDeviceSynchronize();
        thrust::copy( actual_output.begin(), actual_output.end(), layer_outputs.begin() );
        cudaDeviceSynchronize();
        // Propagate through every layer (We need the Actual Output of the Network)
        unsigned int sums_idx = 0;
        unsigned int out_idx = actual_output.size();
        for ( int k = 0; k < w_index_.size(); k++)
        {
            unsigned int w_from = std::get<0>(w_index_[k]);
            unsigned int w_to = std::get<1>(w_index_[k]);
            // Σ( O[j] * W[ji] )
            auto sums = prop_layer( w_from, w_to, actual_output );
            actual_output = sums;
            output_ptr = thrust::raw_pointer_cast( actual_output.data() );          // Is this required?
            cudaDeviceSynchronize();
            // Activate Output: σ( Σ( O[j] * W[ji] ) )
            auto dimA2 = dim1D(actual_output.size());
            sigmoid_activation<<<dimA2.num_blocks_x,dimA2.block_threads_x>>>( output_ptr );
            cudaDeviceSynchronize();
            thrust::copy( actual_output.begin(),actual_output.end(), layer_outputs.begin() + out_idx );
            out_idx += actual_output.size();
            thrust::copy( sums.begin(), sums.end(), layer_sums.begin() + sums_idx );
            sums_idx += sums.size();
            cudaDeviceSynchronize();
        }
        // Copy from Host the Ideal/Target output
        thrust::copy( output.begin()+(output_len*i),output.begin()+(output_len*i)+output_len,ideal.begin());
        cudaDeviceSynchronize();
        // index of output layer's Sums
        unsigned int size_o = layer_sums.size() - output_neurons_;
        // Calculate the Delta nodes of the Output Layer: `-E * σ'(Σ(O[i])`
        auto dimB = dim1D(output_neurons_);
        delta_output<<<dimB.num_blocks_x,dimB.block_threads_x>>>(sums_ptr,ideal_ptr,output_ptr,delta_ptr,size_o);
        // Calculate the Primed Sum: `σ'(Σ[ji])` for all hidden layers
        auto dimC = dim1D(layer_sums.size());
        sigmoid_prime<<<dimC.num_blocks_x,dimC.block_threads_x>>>(sums_ptr,primed_ptr);
        cudaDeviceSynchronize();
        // Calculate the Delta rule for all hidden layers: 
        // k Reversly iterates the weights index (and range), NOT the layer! from last to first hidden (not Input)
        unsigned int size_k = output_neurons_;
        for ( int k = w_index_.size()-1; k > 0; k-- )
        {
            auto w_ik_from = std::get<0>(w_index_[k]);
            auto w_ik_to = std::get<1>(w_index_[k]);
            // size_i =  nodes per hidden layer
            unsigned int size_i = per_layer_;
            // index begins from 1st hidden layer - NOT from input
            unsigned int index = size_i * (k-1);
            float * w_ik = thrust::raw_pointer_cast( weights_.data() ) + w_ik_from;             // up to w_ik_from + w_ik_to
            float * primes  = thrust::raw_pointer_cast( primed_sums.data() ) + index;           // up to index + size_i
            float * delta_i = thrust::raw_pointer_cast( delta_vals.data() ) + index;            // up to index + size_i
            float * delta_k = thrust::raw_pointer_cast( delta_vals.data() ) + index + size_i;   // up to index+size_i+size_k
            // Calculate matrix width
            unsigned int w_size = w_ik_to - w_ik_from;
            unsigned int width = w_size / per_layer_;
            // Temporary storage output - NOTE: Row Major Matrix
            thrust::device_vector<float> mtx_out ( w_size );
            float * mtx_ptr = thrust::raw_pointer_cast( mtx_out.data() );
            // X grid = size_i (Neuron/Node count), Y grid = size_k (Previous Neuron/Node count)
            auto Dim2D = dim2D( size_i, size_k );
            dim3 threadsPerBlock( Dim2D.block_threads_x, Dim2D.block_threads_y );
            dim3 numBlocks( Dim2D.num_blocks_x, Dim2D.num_blocks_y );
            // W[ik] * δ[k]
            delta_product<<<numBlocks,threadsPerBlock>>>( w_ik, delta_k, mtx_ptr, width );
            cudaDeviceSynchronize();
            // Σ( W[ik] * δ[k] ) - Sum Rows of Matrix:  W[ik] * δ[k]
            // NOTE: We save sums in delta_i, we will update their values in the next kernel
            auto dimE = dim1D( size_i );
            delta_sum_rows<<<dimE.num_blocks_x,dimE.block_threads_x>>>( mtx_ptr, delta_i, width );
            cudaDeviceSynchronize();
            // σ'( Σ[ji] ) * Σ( W[ik] * δ[k] )
            delta_hidden<<<dimE.num_blocks_x,dimE.block_threads_x>>>( primes, delta_i ); 
            // Update size_k
            size_k = size_i;
        }
        // Temporary gradients (Row-Major Matrix)
        thrust::device_vector<float> tmp_grads( weights_.size() );
        // Compute (for each δ) the partial derivative `∂E / ∂W[ik]` = δ[k] * O[i]
        // Gradients are equal to weights, Each gradient is allocated to a Weight (W[ik])
        unsigned int delta_idx = 0;
        unsigned int output_idx = 0;
        for ( int k = 0; k < w_index_.size(); k++)
        {
            unsigned int node_deltas = per_layer_;
            unsigned int output_vals = 0;
            // Count Nodes (Output Values)
            if ( k == 0 ) output_vals = input_neurons_;
            else output_vals = per_layer_;
            // Deltas Nodes Exception: last (output) layer *might* have diff delta count
            if ( k == w_index_.size()-1 ) node_deltas = delta_vals.size() - delta_idx;
            float * out_i = thrust::raw_pointer_cast( layer_outputs.data() ) + output_idx;              // output_idx + output_count
            float * delta_k = thrust::raw_pointer_cast( delta_vals.data() ) + delta_idx;                // delta_idx + weight_count
            float * grad_ptr = thrust::raw_pointer_cast( tmp_grads.data() ) + std::get<0>(w_index_[k]); // same index as weights 
            // 2D grid: X is Delta Node count, Y is Ouput Value count, 
            // separate rows by node_deltas (same as weights per node)- create a matrix same as weight matrix
            auto Dim2D = dim2D( node_deltas, output_vals );
            dim3 threadsPerBlock( Dim2D.block_threads_x, Dim2D.block_threads_y );
            dim3 numBlocks( Dim2D.num_blocks_x, Dim2D.num_blocks_y );
            // δ[k] * O[i]
            gradient_descent<<<numBlocks,threadsPerBlock>>>( delta_k, out_i, grad_ptr, node_deltas );
            // Update indexes
            delta_idx += node_deltas;
            output_idx += output_vals;
        }
        cudaDeviceSynchronize();
        // If Online-Training, do back-prop here
        if ( online )
        {
            float * grad_ptr = thrust::raw_pointer_cast( tmp_grads.data() );            // Temp gradients from above loop
            float * weight_ptr = thrust::raw_pointer_cast( weights_.data() );           // All weights
            float * update_ptr = thrust::raw_pointer_cast( updates_.data() );           // All update values
            auto dim_bp = dim1D( weights_.size() );
            back_prop<<<dim_bp.num_blocks_x,dim_bp.block_threads_x>>>( weight_ptr, grad_ptr, update_ptr, alpha_, epsilon_ );
            cudaDeviceSynchronize();
        }
        else
        {
            float * grad_ptr = thrust::raw_pointer_cast( gradients.data() );            // Summed gradients from entire epoch
            float * new_grad = thrust::raw_pointer_cast( tmp_grads.data() );            // Temp gradients from above loop
            auto dim_sum = dim1D( gradients.size() );
            sum_gradients<<<dim_sum.num_blocks_x,dim_sum.block_threads_x>>>( grad_ptr, new_grad );
            cudaDeviceSynchronize();
        }
        // Update Errors for this Epoch: (Ideal - Actual)^2 for each Output node value
        auto dim_err = dim1D( ideal.size() );
        float * error_ptr = thrust::raw_pointer_cast( errors.data() ) + (i*output_neurons_);    // indexed by pattern * output size
        squared_error<<<dim_err.num_blocks_x,dim_err.block_threads_x>>>( ideal_ptr, output_ptr, error_ptr );

    } // End of Epoch Loop

    // if doing BATCH, do back-prop here
    if ( !online )
    {
        float * grad_ptr = thrust::raw_pointer_cast( gradients.data() );                // Summed gradients from entire epoch
        float * weight_ptr = thrust::raw_pointer_cast( weights_.data() );               // All weights
        float * update_ptr = thrust::raw_pointer_cast( updates_.data() );               // All update values
        auto dim_bp = dim1D( weights_.size() );
        back_prop<<<dim_bp.num_blocks_x,dim_bp.block_threads_x>>>( weight_ptr, grad_ptr, update_ptr, alpha_, epsilon_ );   
        cudaDeviceSynchronize();
    }

    float sq_errors = thrust::reduce( errors.begin(), errors.end() );
    return (sq_errors / total);
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
    unsigned int w_size = weights_end - weights_begin;
    unsigned int width = w_size / input.size();
    unsigned int height = input.size();

    const float * input_ptr = thrust::raw_pointer_cast( input.data() );
    const float * weight_ptr = thrust::raw_pointer_cast( weights_.data() ) + weights_begin;

    // Output result - NOTE: Row Major Matrix
    thrust::device_vector<float> mtx( height*width );
    float * mtx_ptr = thrust::raw_pointer_cast( mtx.data() );

    // X grid = matrix rows, Y grid = matrix columns
    auto dimA = dim2D( height, width);
    dim3 threadsPerBlock( dimA.block_threads_x, dimA.block_threads_y );
    dim3 numBlocks( dimA.num_blocks_x, dimA.num_blocks_y );

    // I[j] * W[ji]
    forward_prop<<<numBlocks,threadsPerBlock>>>( weight_ptr, input_ptr, mtx_ptr, width );
    cudaDeviceSynchronize();

    // Store (and Return): Σ(I[j]*W[i])
    thrust::device_vector<float> sums ( width );
    float * sums_ptr = thrust::raw_pointer_cast( sums.data() );

    // Σ(I[j]*W[i]) - Sum the Matrix Columns into a Row Vector - X: Column iterator
    auto dimB = dim1D( width );
    sum_columns<<<dimB.num_blocks_x,dimB.block_threads_x>>>( mtx_ptr, sums_ptr, height, width );
    cudaDeviceSynchronize();

    return sums;
}


};
