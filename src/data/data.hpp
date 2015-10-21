#ifndef _cuANN_data_HPP_
#define _cuANN_data_HPP_
#include "includes.ihh"
namespace cuANN
{

/**
 * @brief a Simple Struct to wrap around an input-output pair
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

    //typedef std::pair < thrust::host_vector<float>,
    //                    thrust::host_vector<float> > row;

    data ( ) = default;

    data ( const std::string filename );

    // TODO: save to file

    // TODO: host_vector to device_vector copying as an operator would be nice

private:

    // a vector of unique pointers to rows - NOTE: maybe use unique_ptr
    std::vector< row > rows_;
    unsigned int num_rows_;
    unsigned int in_vec_size_;
    unsigned int out_vec_size_;
};
}
#endif
