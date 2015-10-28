#ifndef _cuANN_prop_HPP_
#define _cuANN_prop_HPP_
#include "includes.ihh"
namespace cuANN
{
/** 
 * @brief Matrix propagation: vector * matrix dot product
 * @note requires a 2D grid
 * @warning the result is a Matrix, where each row represents the output from multiplying one input with all its weights
 *          the matrix is vectorised using thrust.
 *          the format is: Input[i]*Weight[i]
 */
__global__ void prop_matrix ( 
                              const float * weight, 
                              const float * input, 
                              float * output, 
                              unsigned int w_size, 
                              unsigned int i_size 
                            );

}
#endif
