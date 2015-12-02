namespace cuANN 
{

template <class A> inline
__host__ thrust::host_vector<float> ann::propagate ( 
                                                        A const& func,
                                                        thrust::device_vector<float> & input 
                                                   ) const
{
    if ( input.size() != input_neurons_ )
        throw std::runtime_error( "ann::propagate param input size doesn't match input layer size" );

    // Copy the input on the device memory - we will modify it
    thrust::device_vector<float> output = input;
    float * output_ptr = thrust::raw_pointer_cast( output.data() );

    // Input through activation function
    auto dimA = dim1D(output.size());
    activate<A><<<dimA.num_blocks_x,dimA.block_threads_x>>>(func,output_ptr);

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
        float * output_ptr = thrust::raw_pointer_cast( output.data() );

        // Run the output through the activation function
        auto dimB = dim1D(output.size());
        activate<A><<<dimB.num_blocks_x,dimB.block_threads_x>>>(func,output_ptr);
    }
    return output;
}

template <class A,class D> inline
__host__ float ann::train (
                              A const& func,
                              D const& deriv,
                              cuANN::data & train_data,
                              unsigned int epochs,
                              unsigned int reports,
                              unsigned int max_threads,
                              float stop_error
                          )
{
    // Weight gradients (Row-Major) - NOTE: those are Summed for each Pattern, during each Epoch 
    thrust::device_vector<float> gradients( weights_.size() );
    // The old (previous) Delta Updates `Δw(t-1)`
    thrust::device_vector<float> updates( weights_.size() );
    // Squared Errors (not Mean'ed yet)
    thrust::device_vector<float> sq_errors( train_data.size()*train_data.output_size() );

    // Gradients (Sums) mutex - each thread will update by summing the Pattern's gradients
    std::mutex gradient_mutex;
    
    // trainer data queue
    std::vector< std::shared_ptr<trainer_data> > thread_data;
    
    // Reserve same as MAX_THREADs else we will be doing many copies
    thread_data.reserve(max_threads);
    
    // Allocate thread_data objects - NOTE: Each `trainer_data` must be unique and not shared!
    for (int i = 0; i < max_threads; i++)
    {
        thread_data.push_back( std::make_shared<trainer_data>( weights_, gradients, sq_errors, w_index_, gradient_mutex,
                                                               output_neurons_, input_neurons_, hidden_neurons_, per_layer_ ));
    }

    // trainer thread pool - max threads and thread_data allocated on device memory
    cuANN::trainer_pool thread_pool(max_threads,thread_data);

    // Wait for all allocations to finish.
    cudaDeviceSynchronize();
    float mse = 0.0;

    // Iterate epochs and compare mse
    int k = 0;
    for (int i = 0; i < epochs; i++)
    {
        // Run an epoch and get its MSE: activation func, derivative func, training data, thread pool, thread data
        mse = epoch(func,deriv,train_data,thread_pool,gradients,updates,sq_errors);

        // Report if needed
        if ( k == reports && k != 0 )
        {
            std::cout << "Epoch "<< i << " MSE: " << mse  << std::endl;
            k = 0;
            train_data.shuffle();
        }
        k++;
        if (mse < stop_error) break;
    }
    return mse;
}

template <class A,class D>
__host__ float ann::epoch (
                              A const& func,
                              D const& deriv,
                              const cuANN::data & dataset,
                              cuANN::trainer_pool & thread_pool,
                              thrust::device_vector<float> & gradients,
                              thrust::device_vector<float> & updates,
                              thrust::device_vector<float> & sq_errors
                          )
{
    thread_pool.start();

    // Iterate training set, spawning a thread for each pattern
    for (int i=0; i< dataset.size(); i++)
    {
        // a `trainer` takes as arguments: activation func, deriv func, etc...
        // then submit it to the thread pool
        // thread_pool.submit( ... );
    }
    // When all threads have finished, we can then update the weights and calculate the mean squared errors
    thread_pool.wait();
    thread_pool.stop();


    /* -- NOTE : BELOW IS THE OLD WORKING CODE --
    // TODO: We will only be doing Batch training: 
    // TODO: back_prop will be non-threaded, and will happen once all pattern threads have finished.
    // TODO: We also need to calculate MSE after the loop

    // Do back-prop here
    float * grad_ptr = thrust::raw_pointer_cast( gradients.data() );                // Summed gradients from entire epoch
    float * weight_ptr = thrust::raw_pointer_cast( weights_.data() );               // All weights
    float * update_ptr = thrust::raw_pointer_cast( updates.data() );                // Previous update values
    auto dim_bp = dim1D( weights_.size() );
    back_prop<<<dim_bp.num_blocks_x,dim_bp.block_threads_x>>>( weight_ptr, grad_ptr, update_ptr, alpha_, epsilon_ );   
    cudaDeviceSynchronize();

    // Reduce Squared errors to MSE
    float sq_errors = thrust::reduce( errors.begin(), errors.end() );
    return (sq_errors / total);
    */
    return  0;
}
};
