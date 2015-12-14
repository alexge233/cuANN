namespace cuANN
{
// This is our template implementation file
template <class A,class D>
trainer<A,D>::trainer (
                        A const& func,
                        D const& deriv,
                        unsigned int index
                      )
// #!COPY pattern.input and pattern.output.
//        WARNING: if I use reference the ANN stops learning!
: _func(func), _deriv(deriv), _i(index)
{}

template <class A,class D>
void trainer<A,D>::operator()(std::vector<std::shared_ptr<pattern>> & patterns)
{    
    // Search and find Our Pattern Object
    auto it = std::find_if( patterns.begin(), patterns.end(),
                            [&](const std::shared_ptr<pattern> & ptr)
                            { return ptr->index == _i; });

    // Throw if we can't find our pattern data
    if ( it == patterns.end() ) throw std::runtime_error("didn't find pattern object");

    // Set the trainer data pointer
    const std::shared_ptr<cuANN::pattern> & ptr = (*it);
    assert(ptr);

    // Clear pattern arrays
    ptr->zero_fill();

    // #1 Forward-Propagate: activate input & output through layers
    // `Σ(O[j]*W[i])`: Dot product of previous layer output O[j] by Weight W[i]
    //                 Sum the dot product into a single vector (represents input Sum for layer [i])
    //  And then activate it: `σ'(Σ(I[j]*W[i]))`
    fw_propagate(ptr);

    // #2 Calculate the Primed value of node input sums for all layers
    primed_sums(ptr);

    // #3 Calculate output node delta
    output_node_delta(ptr);

    // NOTE: #2 and #3 can probably be combined in a single kernel: run all primes, and the
    //       If our index is at the end (last layer) also do the delta output

    // #4 Calculate hidden node deltas
    hidden_node_delta(ptr);

    // #5 Calculate the wieght gradients for the entire network
    calc_weight_gradients(ptr);

    // #6 Square Errors
    calc_squared_errors(ptr);
}

template <class A,class D>
void trainer<A,D>::fw_propagate(const std::shared_ptr<pattern> & ptr)
{
    // activate Input and move the activated output to `node_outputs`
    auto dim = dim1D(ptr->input_size);
    activate<A><<<dim.num_blocks_x,dim.block_threads_x>>>(_func,
                                                          thrust::raw_pointer_cast(ptr->ideal_input.data()),
                                                          thrust::raw_pointer_cast(ptr->node_outputs.data())
                                                         );
    unsigned int sums_index = 0;
    unsigned int input_index = 0;

    //Iterate layers
    for (unsigned int k = 0; k < ptr->weight_idx_ref.size(); k++)
    {
        // Find layer's `node_size`
        unsigned int input_size = 0;
        unsigned int output_size = 0;
        if (k == 0){
            input_size = ptr->input_size;
            output_size = ptr->n_per_hl;
        }
        else if (k == ptr->weight_idx_ref.size()-1){
            input_size = ptr->n_per_hl;
            output_size = ptr->output_size;
        }
        else {
            input_size = ptr->n_per_hl;
            output_size = ptr->n_per_hl;
        }

        unsigned int weights_in_layer = std::get<1>(ptr->weight_idx_ref[k]) - std::get<0>(ptr->weight_idx_ref[k]);
        // width is the amount of weights per node = `# of weights in this layer` / `# of nodes in this layer`
        unsigned int width = weights_in_layer / input_size;
        unsigned int height = input_size;

        // X grid = matrix rows / nodes in [j], Y grid = matrix columns / weights in [i]
        auto dimA = dim2D(height,width);
        dim3 tPB(dimA.block_threads_x,dimA.block_threads_y);
        dim3 nB(dimA.num_blocks_x,dimA.num_blocks_y);

        // `O[j] * W[ji]` : @weights, @input, @matrix output, @matrix width - use as input `ptr->node_outputs`
        forward_prop<<<nB,tPB>>>( thrust::raw_pointer_cast(ptr->weight_ref.data()) + std::get<0>(ptr->weight_idx_ref[k]),
                                  thrust::raw_pointer_cast(ptr->node_outputs.data()) + input_index,
                                  thrust::raw_pointer_cast(ptr->fw_prop_mtx.data()),
                                  width);
        
        // I[i] = `Σ(O[j]*W[i])` :  @matrix output, @node sums, @matrix height, @matrix width - save sums output
        // Sum Matrix Columns Into a Row size of(Matrix width )
        auto dB = dim1D(width);
        sum_columns<<<dB.num_blocks_x,dB.block_threads_x>>>(thrust::raw_pointer_cast(ptr->fw_prop_mtx.data()),
                                                            thrust::raw_pointer_cast(ptr->node_sums.data()) + sums_index,
                                                            height,
                                                            width);
        // Update input_index
        input_index += input_size;
        
        // Activate Sums Input I[i]: `σ(Σ(O[j]*W[i]))` - save node output - as `O[i]`
        auto dC = dim1D(width);
        activate<A><<<dC.num_blocks_x,dC.block_threads_x>>>(_func,
                                                            thrust::raw_pointer_cast(ptr->node_sums.data()) + sums_index,
                                                            thrust::raw_pointer_cast(ptr->node_outputs.data()) + input_index );
        // Update Indexes
        sums_index += output_size;
    }
}

template <class A,class D>
void trainer<A,D>::primed_sums(const std::shared_ptr<pattern> & ptr)
{
    // Calculate the Primed Sum: `σ'(Σ[ji])` for all node Input Sums.
    auto dim = dim1D(ptr->node_sums.size());

    // Store the primed Sums in `ptr->primed_sums`
    derivatives<D><<<dim.num_blocks_x,dim.block_threads_x>>>(_deriv,
                                                             thrust::raw_pointer_cast(ptr->node_sums.data()),
                                                             thrust::raw_pointer_cast(ptr->primed_sums.data()));
}

template <class A,class D>
void trainer<A,D>::output_node_delta(const std::shared_ptr<pattern> & ptr)
{
    // index position of output layer's primed sums
    unsigned int size_o = ptr->primed_sums.size() - ptr->output_size;
    unsigned int o_idx = ptr->node_outputs.size() - ptr->output_size;

    // Calculate the Delta nodes of the Output Layer: `-E * σ'(Σ(O[i])`
    auto dim = dim1D(ptr->output_size);
    delta_output<<<dim.num_blocks_x,dim.block_threads_x>>>(thrust::raw_pointer_cast(ptr->primed_sums.data()),
                                                           thrust::raw_pointer_cast(ptr->ideal_output.data()),
                                                           thrust::raw_pointer_cast(ptr->node_outputs.data()) + o_idx,
                                                           thrust::raw_pointer_cast(ptr->node_deltas.data()),
                                                           size_o);
}

template <class A,class D>
void trainer<A,D>::hidden_node_delta(const std::shared_ptr<pattern> & ptr)
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
void trainer<A,D>::calc_weight_gradients(const std::shared_ptr<pattern> &ptr)
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
    // Sum the weight gradients (Batch Training): `Σ( ∂E/∂W[ik] )` - X: gradient index
    auto d1 = dim1D(ptr->epoch_gradients.size());
    sum_gradients<<<d1.num_blocks_x,d1.block_threads_x>>>(thrust::raw_pointer_cast(ptr->epoch_gradients.data()),
                                                          thrust::raw_pointer_cast(ptr->gradients.data()) );
}

template <class A,class D>
void trainer<A,D>::calc_squared_errors(const std::shared_ptr<pattern> &ptr)
{
    // Position of the actual output
    unsigned int o_idx = ptr->node_outputs.size() - ptr->output_size;
    // Index of the global epoch squared errors
    unsigned int e_idx = ptr->output_size * _i;

    // Error = (Ideal - Actual)^2 for each Output node value 
    // We won't lock access to the epoch errors, because we **should not** access other items only this pattern's array range
    auto dim = dim1D(ptr->output_size);
    squared_error<<<dim.num_blocks_x,dim.block_threads_x>>>( thrust::raw_pointer_cast(ptr->ideal_output.data()),
                                                             thrust::raw_pointer_cast(ptr->node_outputs.data()) + o_idx,
                                                             thrust::raw_pointer_cast(ptr->epoch_errors.data()) + e_idx );
}

};
