namespace cuANN
{
// This is our template implementation file
template <class A,class D>
trainer<A,D>::trainer (
                        A const& func,
                        D const& deriv,
                        const cuANN::row & pattern,
                        unsigned int index
                      )
// #!COPY pattern.input and pattern.output.
//        WARNING: if I use reference the ANN stops learning!
: _func(func), _deriv(deriv), ideal_input(pattern.input),ideal_output(pattern.output), _i(index)
{}

template <class A,class D>
void trainer<A,D>::operator()( std::vector<std::shared_ptr<trainer_data>> & thread_data )
{
    // Search and find the first available `trainer_data` object and lock it so we can use it
    auto it = std::find_if( thread_data.begin(), thread_data.end(),
                            [&](const std::shared_ptr<trainer_data> & ptr){ return ptr->available.try_lock(); });

    // Throw if we can't find available thread data - this should never happen if MAX_THREADS == THREAD_DATA
    if ( it == thread_data.end() ) throw std::runtime_error("no free trainer_data");

    /*
         AN interesting Idea would be to fill the GPU memory with trainer-data objects unique to a pattern.
         This scheme, would require that for each pattern, I find the related trainer-data.
         That way, I would use the maximum amount of memory, allocate it once, and avoid a lot of other
         allocations and copying.
    */

    // Set the trainer data pointer
    const std::shared_ptr<cuANN::trainer_data> & ptr = (*it);
    assert(ptr);

    // Reset previous data 
    thrust::fill(thrust::device,ptr->gradients.begin(),ptr->gradients.end(),0.f);
    thrust::fill(thrust::device,ptr->node_sums.begin(),ptr->node_sums.end(),0.f);
    thrust::fill(thrust::device,ptr->node_deltas.begin(),ptr->node_deltas.end(),0.f);
    thrust::fill(thrust::device,ptr->primed_sums.begin(),ptr->primed_sums.end(),0.f);
    thrust::fill(thrust::device,ptr->node_outputs.begin(),ptr->node_outputs.end(),0.f);
    thrust::fill(thrust::device,ptr->actual_output.begin(),ptr->actual_output.end(),0.f);

    // #1 Forward-Propagate: activate input & output through layers
    fw_propagate(ptr);
    
    // #2 Calculate the Primed value of node input sums for all layers
    primed_sums(ptr);

    // #3 Calculate output node delta
    output_node_delta(ptr);

    // NOTE: #2 and #3 can probably be combined in a single kernel: run all primes, and the
    //       If our index is at the end (last layer) also do the delta output
    // 
    // However, the hidden node deltas require than we have first finished primes and output node deltas
    // So, without CUDA Compute 3.5 this can't be done in a single kernel.

    // #4 Calculate hidden node deltas
    hidden_node_delta(ptr);

    // #5  Calculate the wieght gradients for the entire network
    calc_weight_gradients(ptr);

    // #2 & 3, are forward iterations, #4 is reverse, and #5 is forward.
    // Sadly those can't be combined using CUDA compute 3.5, I need a newer GPU and a radical change to the kernel code.

    // #6   (a) Calculate squared errors (for output pattern) and add them to the epoch errors
    calc_squared_errors(ptr);

    // Unlock the available tread_data object
    ptr->available.unlock();
}

template <class A,class D>
void trainer<A,D>::fw_propagate(const std::shared_ptr<trainer_data> & ptr) 
{
    // #!COPY Input as tmp_output (DtoD)
    thrust::device_vector<float> tmp_output(ideal_input.begin(),ideal_input.end());

    // input size = grid x
    auto dim = dim1D(ideal_input.size());

    // Activate and then copy the output to `node_outputs` for delta node calculations
    activate<A><<<dim.num_blocks_x,dim.block_threads_x>>>(_func,thrust::raw_pointer_cast(tmp_output.data()));

    // #!COPY activated input as node_output (can move into the activation function but can't avoid the copy)
    thrust::copy(tmp_output.begin(),tmp_output.end(),ptr->node_outputs.begin() );

    // `sum_idx` = node_sums index, `out_idx` = node_outputs index
    unsigned int sums_idx = 0;
    unsigned int out_idx = tmp_output.size();

    // Propagate through every layer
    for (unsigned int k = 0; k < ptr->weight_idx_ref.size(); k++)
    {
        // Range of weights needed for this layer propagation: [from,to]
        unsigned int w_from = std::get<0>(ptr->weight_idx_ref[k]);
        unsigned int w_to = std::get<1>(ptr->weight_idx_ref[k]);

        // Σ( O[j] * W[ji] ) #!COPY: DtoD (can be avoided if the layer_product is flattened here)
        ptr->layer_sums = trainer<A,D>::layer_product(w_from,w_to,tmp_output,ptr);
        tmp_output = ptr->layer_sums;

        // #!COPY Node Sums (layer_product) into layer_sums (no need, we can directly use `ptr->node_sums`)
        //        BUT it requires that we know the `sums_idx` to correctly copy the sums in place
        thrust::copy(ptr->layer_sums.begin(),ptr->layer_sums.end(),ptr->node_sums.begin()+sums_idx);

        // update sums index: sums_idx + layer_nodes
        sums_idx += tmp_output.size();

        // Activate Output O[i]: `σ(O[i])`
        auto dimA2 = dim1D(tmp_output.size());
        activate<A><<<dimA2.num_blocks_x,dimA2.block_threads_x>>>(_func,thrust::raw_pointer_cast(tmp_output.data()));

        // #!COPY Activated Node output (we can move the copy inside the activation kernel: use sums, and activate them
        //        Moving the values into the `node_outputs`
        thrust::copy(tmp_output.begin(),tmp_output.end(),ptr->node_outputs.begin()+out_idx);

        // update out_idx = out_idx + layer_nodes
        out_idx += tmp_output.size();
    }
        
    // #!COPY moving from local array to the `traner_data` array! Why not use it right from the start then?
    //        In fact, in order to avoid the (Re)allocation of tmp_out at every pattern,
    //        can't I allocate a tmp buffer in `trainer_data` and use it there?
    //        Problem is that tmp_output size CHANGES within the for-loop.
    //        Hence I need a way to track the change in size.
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
    unsigned int w_size = weights_end - weights_begin;
    unsigned int width = w_size / input.size();
    unsigned int height = input.size();

    // X grid = matrix rows, Y grid = matrix columns
    auto dimA = dim2D(height,width);
    dim3 threadsPerBlock(dimA.block_threads_x,dimA.block_threads_y);
    dim3 numBlocks(dimA.num_blocks_x,dimA.num_blocks_y);

    // `I[j] * W[ji]` : @params: weights, input, matrix output, matrix width
    forward_prop<<<numBlocks,threadsPerBlock>>>(thrust::raw_pointer_cast(ptr->weight_ref.data()) + weights_begin,
                                                thrust::raw_pointer_cast(input.data()),
                                                thrust::raw_pointer_cast(ptr->fw_prop_mtx.data()),
                                                width);

    // #!ALLOCATE: `Σ(I[j]*W[i])` - we can use directly the `ptr->node_sums` if we know the correct index!
    //                              Doing so we avoid "#of layers * #of patterns" allocations !
    thrust::device_vector<float> sums (width);

    // `Σ(I[j]*W[i])` :  @params: matrix output, node sums, matrix height, matrix width
    auto dimB = dim1D( width );
    sum_columns<<<dimB.num_blocks_x,dimB.block_threads_x>>>(thrust::raw_pointer_cast(ptr->fw_prop_mtx.data()),
                                                            thrust::raw_pointer_cast(sums.data()),
                                                            height,width);
    // #!COPY is returned by value !!!
    //        This can most certainly be avoided by flattening the `fw_propagate` and moving this code here out!
    return sums;
}

template <class A,class D>
void trainer<A,D>::primed_sums(const std::shared_ptr<trainer_data> & ptr)
{
    // Calculate the Primed Sum: `σ'(Σ[ji])` for all node Input Sums.
    auto dim = dim1D(ptr->node_sums.size());

    // Store the primed Sums in `ptr->primed_sums`
    derivatives<D><<<dim.num_blocks_x,dim.block_threads_x>>>(_deriv,
                                                             thrust::raw_pointer_cast(ptr->node_sums.data()),
                                                             thrust::raw_pointer_cast(ptr->primed_sums.data()));
}

template <class A,class D>
void trainer<A,D>::output_node_delta(const std::shared_ptr<trainer_data> & ptr)
{
    // index position of output layer's primed sums
    unsigned int size_o = ptr->primed_sums.size() - ptr->output_size;

    // Calculate the Delta nodes of the Output Layer: `-E * σ'(Σ(O[i])` - the values are now placed in `ptr->node_deltas`
    auto dim = dim1D(ptr->output_size);
    delta_output<<<dim.num_blocks_x,dim.block_threads_x>>>(thrust::raw_pointer_cast(ptr->primed_sums.data()),
                                                           thrust::raw_pointer_cast(ideal_output.data()),
                                                           thrust::raw_pointer_cast(ptr->actual_output.data()),
                                                           thrust::raw_pointer_cast(ptr->node_deltas.data()),
                                                           size_o);
}

template <class A,class D>
void trainer<A,D>::hidden_node_delta(const std::shared_ptr<trainer_data> & ptr)
{
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

        float * mtx_ptr = thrust::raw_pointer_cast( ptr->node_delta_mtx.data() );

        // NOTE: I can probably combine in a single kernel the following (3) three kernels,
        //       IF I abandon the 2D grid of the delta_product.
        //       Using a 1D grid for iteration of current layer nodes, means that it may be possible
        //       To compute all three kernels in a single one, but there might be a performance hit?

        // X grid = size_i (current layer node count), Y grid = size_k (next layer node count)
        auto Dim2D = dim2D(size_i,size_k);
        dim3 threadsPerBlock( Dim2D.block_threads_x, Dim2D.block_threads_y );
        dim3 numBlocks( Dim2D.num_blocks_x, Dim2D.num_blocks_y );

        // W[ik] * δ[k]
        delta_product<<<numBlocks,threadsPerBlock>>>(w_ik,delta_k,mtx_ptr,width);

        // Σ( W[ik] * δ[k] ) - Sum Rows of Matrix:  W[ik] * δ[k]
        // WARNING: We STORE sums in `delta_i`!
        auto dim = dim1D(size_i);
        delta_sum_rows<<<dim.num_blocks_x,dim.block_threads_x>>>(mtx_ptr,delta_i,width);

        // σ'( Σ[ji] ) * Σ( W[ik] * δ[k] ) - `delta_i` must already contain the Sum Rows: `Σ( W[ik] * δ[k] )`
        delta_hidden<<<dim.num_blocks_x,dim.block_threads_x>>>(primes,delta_i);

        // Update size_k
        size_k = size_i;
    }  
}

template <class A,class D>
void trainer<A,D>::calc_weight_gradients(const std::shared_ptr<trainer_data> &ptr)
{
    // Index of node delta per layer
    unsigned int delta_idx = 0;

    // Index of node outputs per layer
    unsigned int output_idx = 0;

    // NOTE: This loop could probably be moved into a single kernel
    //       If I can pass to that kernel the actual weight index (as an array or arrays).
    //       This would imply that instead of a 2D grid, I flatten it to a 1D grid, and iterate weights
    //       Instead of Nodes & Weights ?
    //       Alternatively, it may be possible to do as a 2D grid, If I can index weights, gradients and node deltas properly
    //       However, the benefit would be small: I would just avoid a few kernel calls.

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

    // Sum the weight gradients (Batch Training): `Σ( ∂E/∂W[ik] )` - X: gradient index
    auto dim = dim1D(ptr->epoch_gradients.size());
    sum_gradients<<<dim.num_blocks_x,dim.block_threads_x>>>(thrust::raw_pointer_cast(ptr->epoch_gradients.data()),
                                                            thrust::raw_pointer_cast(ptr->gradients.data()) );
}

template <class A,class D>
void trainer<A,D>::calc_squared_errors(const std::shared_ptr<trainer_data> &ptr)
{
    auto dim = dim1D(ptr->output_size);

    // indexed by pattern index, up to index + output size
    unsigned int index = _i*ptr->output_size;

    // Error = (Ideal - Actual)^2 for each Output node value 
    // We won't lock access to the epoch errors, because we **should not** access other items only this pattern's array range
    squared_error<<<dim.num_blocks_x,dim.block_threads_x>>>( thrust::raw_pointer_cast(ideal_output.data()),
                                                             thrust::raw_pointer_cast(ptr->actual_output.data()),
                                                             thrust::raw_pointer_cast(ptr->epoch_errors.data())+index);
}

};
