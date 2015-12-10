#include "ann.hpp"

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

    // See how to calculate initiali weight values: 
    // http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
    // `( -1/√d, 1/√d )` where d is the number of input nodes.
    float upper = 1.0 / std::sqrt( input_neurons_ );
    float lower = -1.0 / std::sqrt( input_neurons_ );

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

__host__ void ann::print_weights() const
{
    if ( weights_.size() > 0 )
    {
        for ( int k = 0; k < w_index_.size(); k++)
        {
            auto from = std::get<0>(w_index_[k]);
            auto to   = std::get<1>(w_index_[k]);
            for ( int x = from; x < to; x++ )
                std::cout << weights_[x] << " ";
            std::cout << std::endl;
        }
    }
    else 
        throw std::runtime_error("cannot print weights - they are null");
}
 
__host__ thrust::device_vector<float> ann::prop_layer ( 
                                                          unsigned int weights_begin,
                                                          unsigned int weights_end,
                                                          const thrust::device_vector<float> & input
                                                      ) const
{
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
