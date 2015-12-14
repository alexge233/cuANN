namespace cuANN 
{

template <class A>
thrust::device_vector<float> ann::propagate ( 
                                                A const& func,
                                                thrust::device_vector<float> & input 
                                           ) const
{
    if ( input.size() != input_neurons_ )
        throw std::runtime_error( "ann::propagate param input size doesn't match input layer size" );

    // #!COPY: move from host to device memory - we will modify it
    thrust::device_vector<float> output = input;
    float * output_ptr = thrust::raw_pointer_cast(output.data());

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
        cudaDeviceSynchronize();

        // Copy the sums as output
        output = sums;
        float * output_ptr = thrust::raw_pointer_cast( output.data() );
        cudaDeviceSynchronize();
        
        // Run the output through the activation function
        auto dimB = dim1D(output.size());
        activate<A><<<dimB.num_blocks_x,dimB.block_threads_x>>>(func,output_ptr);
        cudaDeviceSynchronize();
    }
    return output;
}

template <class A,class D> 
float ann::train (
                      A const& func,
                      D const& deriv,
                      cuANN::data & train_data,
                      unsigned int epochs,
                      unsigned int reports,
                      unsigned int max_threads, // DEPRECATE
                      float stop_error,
                      float learning,
                      float momentum
                  )
{
    std::cout << "Epochs: " << epochs << " CPU Threads: " << max_threads << " Stop-Error: " << stop_error << std::endl;
    std::cout << "Learning ε: " << learning << " Momentum α: " << momentum << std::endl;
    alpha_ = momentum; 
    epsilon_ = learning;

    // Weight gradients (Row-Major) - NOTE: those are Summed for each Pattern, during each Epoch 
    thrust::device_vector<float> gradients( weights_.size() );
    // The old (previous) Delta Updates `Δw(t-1)`
    thrust::device_vector<float> updates( weights_.size() );
    // Epoch Squared Errors: One for each Output Node * Patterns
    thrust::device_vector<float> errors(train_data.size()*train_data.output_size());
    // trainer pattern data 
    std::vector<std::shared_ptr<pattern>> patterns;
    patterns.reserve(train_data.size());
    // Allocate pattern objects on GPU memory
    // WARNING: I should check for available memory before actually creating a new `pattern` object!
    for (int i = 0; i < train_data.size(); i++)
    {
        patterns.push_back( std::make_shared<pattern>( train_data[i].input, train_data[i].output,
                                                        weights_, gradients, errors, w_index_, 
                                                        output_neurons_, input_neurons_, hidden_neurons_, per_layer_, 
                                                        i ));
    }
    cudaDeviceSynchronize();

    float mse = 0.0;
    // Iterate epochs and compare mse
    int k = 0;
    for (int i = 0; i < epochs; i++)
    {
        // Run an epoch and get its MSE
        mse = epoch(func,deriv,train_data.size(),patterns,gradients,updates,errors);

        // Report if needed
        if ( k == reports && k != 0 )
        {
            std::cout << "Epoch " << i << " MSE: " << mse  << std::endl;
            k = 0;
            //std::cout << "Test MSE: " << test(func,train_data) << std::endl;
        }
        k++;
        if (mse < stop_error) break;
    }
    return mse;
}

template <class A>
float ann::test (
                    A const& func,
                    const cuANN::data & test_data
                ) const
{
    // Epoch Squared Errors (not Mean'ed yet)
    thrust::device_vector<float> errors(test_data.size() * test_data.output_size());
    thrust::device_vector<float> input(test_data.input_size()); 
    thrust::device_vector<float> ideal(test_data.output_size());
    thrust::device_vector<float> output(test_data.output_size());
    cudaDeviceSynchronize();
    // Iterate testing set, propagating each input
    for (unsigned int i = 0; i < test_data.size(); i++)
    {
        input = thrust::device_vector<float>(test_data[i].input);
        ideal = thrust::device_vector<float>(test_data[i].output);
        output = propagate(func,input);
        int idx = i * test_data.output_size();
        // Compute squared error
        auto dim = dim1D(test_data.output_size());
        squared_error<<<dim.num_blocks_x,dim.block_threads_x>>>(thrust::raw_pointer_cast(ideal.data()),
                                                                thrust::raw_pointer_cast(output.data()),
                                                                thrust::raw_pointer_cast(errors.data())+idx);
    }
    float sum = thrust::reduce(errors.begin(),errors.end());
    cudaDeviceSynchronize();
    return sum / (test_data.size()*test_data.output_size());
}

template <class A,class D>
float ann::epoch (
                   A const& func,
                   D const& deriv,
                   const unsigned int datasize,
                   std::vector<std::shared_ptr<pattern>> & patterns,
                   thrust::device_vector<float> & gradients,
                   thrust::device_vector<float> & updates,
                   thrust::device_vector<float> & errors
                 )
{
    // WARNING: We need to zero-fill the gradients, else we are summing gradients for all epochs!
    thrust::fill(gradients.begin(),gradients.end(),0.f);
    thrust::fill(errors.begin(),errors.end(),0.f);
    cudaDeviceSynchronize();

    // random pool of numbers from 0 ~ datasize
    std::vector<int> rand_pool(datasize);
    std::iota(rand_pool.begin(),rand_pool.end(),0);
    // randomly shuffle
    auto prng = std::default_random_engine{};
    prng.seed( std::random_device{}() );
    std::shuffle( std::begin(rand_pool), std::end(rand_pool), prng);

    // at this point we're training on a randomly chosen pattern
    for (unsigned int i = 0; i < rand_pool.size(); i++)
    {
        // Create a trainer and run it
        auto obj = cuANN::trainer<A,D>(func,deriv,i);
        obj(patterns);
    }
    
    // Do back-prop here
    auto dim = dim1D( weights_.size() );
    back_prop<<<dim.num_blocks_x,dim.block_threads_x>>>( thrust::raw_pointer_cast(weights_.data()), 
                                                         thrust::raw_pointer_cast(gradients.data()), 
                                                         thrust::raw_pointer_cast(updates.data()), 
                                                         alpha_, 
                                                         epsilon_);
    // Copy squared errors to host
    thrust::host_vector<float> sq_errors(errors);
    cudaDeviceSynchronize();
    // Sum squared errors for all patterns
    float sum = thrust::reduce(sq_errors.begin(),sq_errors.end());
    float mse = sum / sq_errors.size();

    /*
    std::cout << "squared errors: ";
    for (const auto & v : sq_errors)
        std::cout << v << " ";
    std::cout << std::endl << "MSE: " << mse << std::endl;
    */

    // Return Mean-Squared Errors
    return mse;
}

template<class Archive> 
void ann::save(Archive & ar, const unsigned int version) const
{
    // Serialize `weights` which is a `thrust::device_vector<float>` by copying to an `std::vector` first
    std::vector<float> tmp_weights;

    // Copy one by one the weights from the device memory
    for(int i = 0; i < weights_.size(); i++) tmp_weights.push_back(weights_[i]);
    ar & tmp_weights;

    ar & input_neurons_;
    ar & hidden_neurons_;
    ar & hidden_layers_;
    ar & per_layer_;
    ar & w_index_;
}

template<class Archive>
void ann::load(Archive & ar, const unsigned int version)
{
    // Last part of the archive are our `std::vector<float>` weights
    std::vector<float> tmp_weights;
    ar & tmp_weights;

    // Copy one by one the weights to the device memory
    weights_ = thrust::device_vector<float>(tmp_weights.size());
    for(int i = 0; i < tmp_weights.size(); i++) weights_[i] = tmp_weights[i];

    ar & input_neurons_;
    ar & hidden_neurons_;
    ar & hidden_layers_;
    ar & per_layer_;
    ar & w_index_;
}

};
