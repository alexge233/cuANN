#ifndef _cuANN_data_HPP_
#define _cuANN_data_HPP_
#include "includes.ihh"
namespace cuANN
{

/**
 * @brief a Simple Struct to wrap around an input-output pair
 * NOTE - One bottle-neck is that row uses a host_vector
 *        If only I could load all training data to the device
 *        and keep it there for good.
 */
struct row
{
    thrust::host_vector<float> input;
    thrust::host_vector<float> output;

    row ( unsigned int in_size, unsigned int out_size );
};

/**
 * @brief Struct wrapper around rows of data
 * @class data
 * @date 21th October 2015
 * @author Alex Giokas <alexge233@hotmail.com>
 * @version 1
 *
 * The Data format is identical to that used by lib FANN
 * 1st line of the file must contain: number of entries, input count, output count
 * e.g.,: `500 2 1`
 * This implies that we have 1000 rows: 500 inputs and 500 outputs, 
 * where the input is 2 neurons, and the output is one neuron.
 * Each line should have a UNIX carriage return: `\r\n`
 */
class data
{
public:
   
    /// No empty constructor
    data ( ) = delete;

    /// Load data from disk
    data ( const std::string filename );

    /// Get the data size
    int size() const;

    /// Expose const iterators to our data's rows
    using const_iterator = std::vector<row>::const_iterator;

    const_iterator begin() const;

    const_iterator end() const;

    /// TODO: I need a way to load all rows into device memory
    ///       and keep them there during training

private:

    // a vector of unique pointers to rows - NOTE: maybe use unique_ptr
    std::vector< row > rows_;
    unsigned int num_rows_;
    unsigned int in_vec_size_;
    unsigned int out_vec_size_;
};
}
#endif
