namespace cuANN
{
// This is our template implementation file
template <class A,class D>
trainer<A,D>::trainer (
                        A const& func,
                        D const& deriv,
                        const std::shared_ptr<cuANN::trainer_data> trainer_data,
                        const thrust::host_vector<float> & input,
                        const thrust::host_vector<float> & output,
                        float alpha,
                        float epsilon,
                        unsigned int index
                      )
: _dmem(trainer_data), _a(alpha), _e(epsilon), _i(index), _func(func), _deriv(deriv)
{
    assert(trainer_data && _dmem);
    
    // copy input
    thrust::copy(input.begin(),input.end(),_dmem->input.begin());

    // copy ideal output
    thrust::copy(output.begin(),output.end(),_dmem->ideal_out.begin());
    
    // Fill with zeros: node_sums, node_deltas, primed_sums, nodes_output, gradients, sq_errors 
    thrust::fill(thrust::device,_dmem->node_sums.begin(),_dmem->node_sums.end(),0.f);
    thrust::fill(thrust::device,_dmem->node_deltas.begin(),_dmem->node_deltas.end(),0.f);
    thrust::fill(thrust::device,_dmem->primed_sums.begin(),_dmem->primed_sums.end(),0.f);
    thrust::fill(thrust::device,_dmem->nodes_output.begin(),_dmem->nodes_output.end(),0.f);
    thrust::fill(thrust::device,_dmem->gradients.begin(),_dmem->gradients.end(),0.f);
    thrust::fill(thrust::device,_dmem->sq_errors.begin(),_dmem->sq_errors.end(),0.f);

    // wait for GPU to finish all memory allocations and zero-fills
    cudaDeviceSynchronize();
}

template <class A,class D>
void trainer<A,D>::operator()( std::vector<std::shared_ptr<trainer_data>> & thread_data ) const
{
    // Search and find the first available `trainer_data` object and lock it so we can use it
    auto it = std::find_if( thread_data.begin(), thread_data.end(),
                            [&](const std::shared_ptr<trainer_data> & ptr){ return ptr->available.try_lock(); });

    // Throw if we can't find available thread data - this should never happen if MAX_THREADS == THREAD_DATA
    if ( it == thread_data.end() ) throw std::runtime_error("no free trainer_data");

    // #1 Forward-Propagate Input Pattern:
    //    (a) Activate input
    //    (b)  Propagate Node output from each layer as next layer's input
    //    (c) Store all node output & all node input sums

    // #2 Calculate Output Node Delta

    // #3 Calculate the Primed value of node input sums for all layers

    // #4 Reversely calculate the Node Delta for all hidden layers

    // #5   (a) Calculate the Gradient values for the entire network
    //      (b) Sum gradients (lockable mutex) for the epoch to which this `trainer` belongs to

    // #6   (a) Calculate squared errors (for output pattern)
    //      (b) add them (lockable mutex) to the global errrors

    // NOTE: #2 and #3 can run in parallel after #1 and before #4
    // NOTE: #6 can run in parallel after #1 is complete
}

};

/* -- NOTE : BELOW IS THE OLD WORKING CODE --

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

    float * grad_ptr = thrust::raw_pointer_cast( gradients.data() );            // Summed gradients from entire epoch
    float * new_grad = thrust::raw_pointer_cast( tmp_grads.data() );            // Temp gradients from above loop
    auto dim_sum = dim1D( gradients.size() );
    sum_gradients<<<dim_sum.num_blocks_x,dim_sum.block_threads_x>>>( grad_ptr, new_grad );
    cudaDeviceSynchronize();

} // End of Epoch Loop

*/

