#ifndef _cuANN_prop_HPP_
#define _cuANN_prop_HPP_
#include "includes.ihh"
namespace cuANN
{
/** 
 * @brief Flat propagation through a vector to a vector
 * @note requires a 1D grid
 * @note This kernal Sums and Computes the sigmnoid Activation in a single call
 */
__global__ void prop_vector ( 
                               float * weight, 
                               float * input, 
                               float * output, 
                               unsigned int size 
                            );

/** 
 * @brief Matrix propagation, from two vectors into a matrix (vectorised)
 * @note requires a 2D grid
 * @warning the result is a Matrix, where each row is all the values of a Weight * Inpuit
 *          thus the columns represent each Weight for a specific layer
 * 
 * @note this kernel does NOT Sum, nor compute the Sigmoid.
 *       that needs be done at a later point.
 */
__global__ void prop_matrix ( 
                              float * weight, 
                              float * input, 
                              float * output, 
                              unsigned int w_size, 
                              unsigned int i_size 
                            );

}
#endif
