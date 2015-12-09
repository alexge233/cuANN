#ifndef _cuANN_data_HPP_
#define _cuANN_data_HPP_
#include "includes.ihh"
namespace cuANN
{
/// @brief An Input/Output Pattern pair
struct row
{
    // The ANN trainer algorithm BUGs out when I use device_vector here!
    //thrust::device_vector<float> input;
    //thrust::device_vector<float> output;
    thrust::host_vector<float> input;
    thrust::host_vector<float> output;

    row ( unsigned int in_size, unsigned int out_size );
};
///
/// @author Alexander Giokas <a.gkiokas@warwick.ac.uk>
/// @date November 2015
/// @class data
/// @brief Struct wrapper around rows of data
///
/// The Data format is identical to that used by lib FANN
/// 1st line of the file must contain: number of entries, input count, output count
/// e.g.,: `500 2 1`
/// This implies that we have 1000 rows: 500 inputs and 500 outputs, 
/// where the input is 2 neurons, and the output is one neuron.
/// Each line should have a UNIX carriage return: `\r\n`
///
class data
{
public:
    /// const iterators to our data's rows
    using const_iterator = std::vector<row>::const_iterator;  

    /// No empty constructor
    data ( ) = delete;
    /// Load data from disk
    data ( const std::string filename );
    /// Get the data size
    int size() const;
    /// Get the Input vector size
    int input_size() const;
    /// Get the Output vector size
    int output_size() const;
    /// Subscript Operator
    const row & operator [] ( const int idx ) const;
    /// Iterate begin
    const_iterator begin() const;
    /// Iterate end
    const_iterator end() const;
    /// Random shuffling of row patterns/pairs
    void shuffle ();
    /// Print on stdout
    void print () const;

private:

    std::vector<row> rows_;
    unsigned int num_rows_;
    unsigned int in_vec_size_;
    unsigned int out_vec_size_;
};
}
#endif
