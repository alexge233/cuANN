namespace cuANN
{
// This is our template implementation file
template <class A,class D>
trainer<A,D>::trainer (
                        A const& func,
                        D const& deriv,
                        const thrust::host_vector<float> & input,
                        const thrust::host_vector<float> & output,
                        float alpha,
                        float epsilon,
                        unsigned int index
                      )
: _func(func), _deriv(deriv), _input(input),ideal_output(output),
  _a(alpha), _e(epsilon), _i(index)
{}

template <class A,class D>
void trainer<A,D>::operator()( std::vector<std::shared_ptr<trainer_data>> & thread_data )
{
    // Search and find the first available `trainer_data` object and lock it so we can use it
    auto it = std::find_if( thread_data.begin(), thread_data.end(),
                            [&](const std::shared_ptr<trainer_data> & ptr){ return ptr->available.try_lock(); });
    // Throw if we can't find available thread data - this should never happen if MAX_THREADS == THREAD_DATA
    if ( it == thread_data.end() ) throw std::runtime_error("no free trainer_data");

    // Set the trainer data pointer
    const std::shared_ptr<cuANN::trainer_data> & ptr = (*it);

    // copy input - host to device
    thrust::copy(_input.begin(),_input.end(),ptr->input.begin());

    // Reset previous data - NOTE: maybe not all of them are needed - I think only the kernel which use previous values
    thrust::fill(thrust::device,ptr->gradients.begin(),ptr->gradients.end(),0.f);
    thrust::fill(thrust::device,ptr->node_sums.begin(),ptr->node_sums.end(),0.f);
    thrust::fill(thrust::device,ptr->node_deltas.begin(),ptr->node_deltas.end(),0.f);
    thrust::fill(thrust::device,ptr->primed_sums.begin(),ptr->primed_sums.end(),0.f);
    thrust::fill(thrust::device,ptr->node_outputs.begin(),ptr->node_outputs.end(),0.f);
    thrust::fill(thrust::device,ptr->actual_output.begin(),ptr->actual_output.end(),0.f);

    // #1 Forward-Propagate: activate input & output through layers
    fw_propagate(ptr);

    // #2 Calculate output node delta
    output_node_delta(ptr);

    // #3 Calculate the Primed value of node input sums for all layers
    primed_sums(ptr);

    // #4 Calculate hidden node deltas
    hidden_node_delta(ptr);

    // #5  Calculate the wieght gradients for the entire network
    calc_weight_gradients(ptr);

    // #6   (a) Calculate squared errors (for output pattern) and add them to the epoch errors
    calc_squared_errors(ptr);

    // Unlock the available tread_data object
    ptr->available.unlock();
}

template <class A,class D>
void trainer<A,D>::fw_propagate(const std::shared_ptr<trainer_data> & ptr) 
{
    assert(ptr);
    // Copy the Input as tmp_output
    thrust::device_vector<float> tmp_output( ptr->input.begin(), ptr->input.end() ); // TODO: allocate array on `trainer_data`

    // input size = grid x
    auto dim = dim1D(ptr->input.size());
    // Activate and then copy the output to `node_outputs` for delta node calculations
    activate<A><<<dim.num_blocks_x,dim.block_threads_x>>>(_func,thrust::raw_pointer_cast(tmp_output.data()));
    thrust::copy(tmp_output.begin(),tmp_output.end(),ptr->node_outputs.begin() );
    cudaDeviceSynchronize();

    // `sum_idx` = node_sums index, `out_idx` = node_outputs index
    unsigned int sums_idx = 0;
    unsigned int out_idx = tmp_output.size();
    // Propagate through every layer
    for (unsigned int k = 0; k < ptr->weight_idx_ref.size(); k++)
    {
        // Range of weights needed for this layer propagation: [from,to]
        unsigned int w_from = std::get<0>(ptr->weight_idx_ref[k]);
        unsigned int w_to = std::get<1>(ptr->weight_idx_ref[k]);

        // Σ( O[j] * W[ji] ) - TODO: allocate `sums` to the device `trainer_data`
        thrust::device_vector<float> sums = trainer<A,D>::layer_product(w_from,w_to,tmp_output,ptr);
        tmp_output = sums; 
        cudaDeviceSynchronize();

        // Copy Node Sums
        thrust::copy(sums.begin(),sums.end(),ptr->node_sums.begin()+sums_idx);
        sums_idx += tmp_output.size();

        // Activate Output O[i]: `σ(O[i])`
        auto dimA2 = dim1D(tmp_output.size());
        activate<A><<<dimA2.num_blocks_x,dimA2.block_threads_x>>>(_func,thrust::raw_pointer_cast(tmp_output.data()));

        // Copy Nodes output (activated)
        thrust::copy(tmp_output.begin(),tmp_output.end(),ptr->node_outputs.begin()+out_idx);
        out_idx += tmp_output.size();
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
        
    // Copy as output the tmp_output
    ptr->actual_output = tmp_output;
}

template <class A,class D>
thrust::device_vector<float> trainer<A,D>::layer_product ( 
                                                            unsigned int weights_begin,
                                                            unsigned int weights_end,
                                                            const thrust::device_vector<float> & input,
                                                            const std::shared_ptr<trainer_data> & ptr
                                                         )
{
    assert(ptr);
    unsigned int w_size = weights_end - weights_begin;
    unsigned int width = w_size / input.size();
    unsigned int height = input.size();

    // Output result - NOTE: Row Major Matrix - Stores `I[j] * W[ji]` - TODO: Allocate space in `trainer_data` for this!!!
    thrust::device_vector<float> mtx( height*width );

    // X grid = matrix rows, Y grid = matrix columns
    auto dimA = dim2D(height,width);
    dim3 threadsPerBlock(dimA.block_threads_x,dimA.block_threads_y);
    dim3 numBlocks(dimA.num_blocks_x,dimA.num_blocks_y);

    // `I[j] * W[ji]` : @params: weights, input, matrix output, matrix width
    forward_prop<<<numBlocks,threadsPerBlock>>>(thrust::raw_pointer_cast(ptr->weight_ref.data()) + weights_begin,
                                                thrust::raw_pointer_cast(input.data()),
                                                thrust::raw_pointer_cast(mtx.data()),
                                                width);
    // Store (and Return): Σ(I[j]*W[i])
    thrust::device_vector<float> sums (width);

    // `Σ(I[j]*W[i])` :  @params: matrix output, node sums, matrix height, matrix width
    auto dimB = dim1D( width );
    sum_columns<<<dimB.num_blocks_x,dimB.block_threads_x>>>(thrust::raw_pointer_cast(mtx.data()),
                                                            thrust::raw_pointer_cast(sums.data()),
                                                            height,width);
    return sums;
}

template <class A,class D>
void trainer<A,D>::output_node_delta(const std::shared_ptr<trainer_data> & ptr)
{
    assert(ptr);
    // index position of output layer node sums
    unsigned int size_o = ptr->node_sums.size() - ptr->output_size;
    // Copy from Host to Device
    thrust::device_vector<float> ideal(ideal_output);           // TODO: allocate `ideal_output` on `trainer_data`
    // Calculate the Delta nodes of the Output Layer: `-E * σ'(Σ(O[i])` - the values are now placed in `ptr->node_deltas`
    // Use template <D> for the derivative, and pass it too
    auto dim = dim1D(ptr->output_size);
    delta_output<D><<<dim.num_blocks_x,dim.block_threads_x>>>(_deriv,
                                                              thrust::raw_pointer_cast(ptr->node_sums.data()),
                                                              thrust::raw_pointer_cast(ideal.data()),
                                                              thrust::raw_pointer_cast(ptr->actual_output.data()),
                                                              thrust::raw_pointer_cast(ptr->node_deltas.data()),
                                                              size_o);
}

template <class A,class D>
void trainer<A,D>::primed_sums(const std::shared_ptr<trainer_data> & ptr)
{
    assert(ptr);
    // Calculate the Primed Sum: `σ'(Σ[ji])` for all hidden layers
    auto dim = dim1D(ptr->node_sums.size());
    // Store the primed Sums in `ptr->primed_sums`
    derivatives<D><<<dim.num_blocks_x,dim.block_threads_x>>>(_deriv,
                                                             thrust::raw_pointer_cast(ptr->node_sums.data()),
                                                             thrust::raw_pointer_cast(ptr->primed_sums.data()));
}

template <class A,class D>
void trainer<A,D>::hidden_node_delta(const std::shared_ptr<trainer_data> & ptr)
{
    assert(ptr);

    // `size_k` is the previous node code (next layer node count)
    unsigned int size_k = ptr->output_size;

    // Iterate from Last Hidden to First Hidden - Skip Output and Input Layers
    // k Reversly iterates the weights index and range (but not layer which must be deduced)
    for (unsigned int k = ptr->weight_idx_ref.size()-1; k > 0; k-- )
    {
        // Get Weight Ranges
        auto w_ik_from = std::get<0>(ptr->weight_idx_ref[k]);
        auto w_ik_to = std::get<1>(ptr->weight_idx_ref[k]);

        // `size_i` =  current layer node count
        unsigned int size_i = ptr->n_per_hl;

        // index begins from 1st hidden layer - not from input
        unsigned int index = size_i * (k-1);

        const float * w_ik = thrust::raw_pointer_cast(ptr->weight_ref.data())+w_ik_from;
        float * primes  = thrust::raw_pointer_cast(ptr->primed_sums.data())+index;
        float * delta_i = thrust::raw_pointer_cast(ptr->node_deltas.data())+index;
        float * delta_k = thrust::raw_pointer_cast(ptr->node_deltas.data())+index+size_i;

        // matrix width: range / hidden nodes per layer = weights per node
        unsigned int w_size = w_ik_to - w_ik_from;
        unsigned int width = w_size / ptr->n_per_hl;

        // Temp storage output - Row Major Matrix - Store: `W[ik]*δ[k]`
        thrust::device_vector<float> mtx_out ( w_size );
        float * mtx_ptr = thrust::raw_pointer_cast( mtx_out.data() );

        // X grid = size_i (current layer node count), Y grid = size_k (next layer node count)
        auto Dim2D = dim2D(size_i,size_k);
        dim3 threadsPerBlock( Dim2D.block_threads_x, Dim2D.block_threads_y );
        dim3 numBlocks( Dim2D.num_blocks_x, Dim2D.num_blocks_y );

        // W[ik] * δ[k]
        delta_product<<<numBlocks,threadsPerBlock>>>(w_ik,delta_k,mtx_ptr,width);
        cudaDeviceSynchronize();

        // Σ( W[ik] * δ[k] ) - Sum Rows of Matrix:  W[ik] * δ[k]
        // WARNING: We STORE sums in `delta_i`!
        auto dim = dim1D(size_i);
        delta_sum_rows<<<dim.num_blocks_x,dim.block_threads_x>>>(mtx_ptr,delta_i,width);
        cudaDeviceSynchronize();

        // σ'( Σ[ji] ) * Σ( W[ik] * δ[k] ) - `delta_i` must already contain the Sum Rows: `Σ( W[ik] * δ[k] )`
        delta_hidden<<<dim.num_blocks_x,dim.block_threads_x>>>(primes,delta_i);
        cudaDeviceSynchronize();

        // Update size_k
        size_k = size_i;
    }  
}

template <class A,class D>
void trainer<A,D>::calc_weight_gradients(const std::shared_ptr<trainer_data> &ptr)
{
    assert(ptr);

    // Index of node delta per layer
    unsigned int delta_idx = 0;
    // Index of node outputs per layer
    unsigned int output_idx = 0;

    // Compute for each node delta the partial derivative `∂E/∂W[ik]` = δ[k] * O[i]
    // Gradients are equal to weights, Each gradient is allocated to a Weight (W[ik])
    for (unsigned int k = 0; k < ptr->weight_idx_ref.size(); k++)
    {
        // node delta count for this layer (same as nodes per layer)
        unsigned int node_deltas = ptr->n_per_hl;
        // last (output) layer may have different delta count
        if ( k == ptr->weight_idx_ref.size()-1 ) 
            node_deltas = ptr->node_deltas.size()-delta_idx;

        // Layer node count (same as layer output values)
        unsigned int output_vals = 0;
        if ( k == 0 ) output_vals = ptr->input_size;
        else output_vals = ptr->n_per_hl;
        
        // gradient values will be updated from this position
        unsigned int gradient_idx = std::get<0>(ptr->weight_idx_ref[k]);
        
        // 2D grid: X is Delta Node count, Y: layer node count
        // X: rows height = node_deltas (same as weights per node)
        auto Dim2D = dim2D( node_deltas, output_vals );
        dim3 threadsPerBlock( Dim2D.block_threads_x, Dim2D.block_threads_y );
        dim3 numBlocks( Dim2D.num_blocks_x, Dim2D.num_blocks_y );

        // Calculate Gradients: `∂E/∂W[ik]= δ[k]*O[i]` - Input: next layer node deltas `δ[k]`, this layer node outputs `O[i]`
        gradient_descent<<<numBlocks,threadsPerBlock>>>( thrust::raw_pointer_cast(ptr->node_deltas.data())+delta_idx,
                                                         thrust::raw_pointer_cast(ptr->node_outputs.data())+output_idx, 
                                                         thrust::raw_pointer_cast(ptr->gradients.data())+gradient_idx,
                                                         node_deltas );
        // Update indexes
        delta_idx += node_deltas;
        output_idx += output_vals;
    }

    // Lock the gradient Sums Mutex - only one thread at a time may update it
    std::lock_guard<std::mutex> guard(ptr->grad_sums_mtx);
    cudaDeviceSynchronize();
    
    // Sum the weight gradients (Batch Training): `Σ( ∂E/∂W[ik] )` - X: gradient index
    auto dim = dim1D(ptr->epoch_gradients.size());
    sum_gradients<<<dim.num_blocks_x,dim.block_threads_x>>>(thrust::raw_pointer_cast(ptr->epoch_gradients.data()),
                                                            thrust::raw_pointer_cast(ptr->gradients.data()) );
}

template <class A,class D>
void trainer<A,D>::calc_squared_errors(const std::shared_ptr<trainer_data> &ptr)
{
    assert(ptr);
    thrust::device_vector<float> ideal(ideal_output);        // TODO: allocate `ideal_output` on `trainer_data`
    auto dim = dim1D(ideal.size());
    // indexed by pattern index * output size
    unsigned index = _i * ptr->output_size; 

    cudaDeviceSynchronize();
    // Error = (Ideal - Actual)^2 for each Output node value 
    // We won't lock access to the epoch errors, because we **should not** access other items only this pattern's array range
    squared_error<<<dim.num_blocks_x,dim.block_threads_x>>>( thrust::raw_pointer_cast(ideal.data()), 
                                                             thrust::raw_pointer_cast(ptr->actual_output.data()),
                                                             thrust::raw_pointer_cast(ptr->epoch_errors.data())+index);
}

};
