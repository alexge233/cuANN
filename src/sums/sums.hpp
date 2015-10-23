#ifndef _cuANN_sums_HPP_
#define _cuANN_sums_HPP_
#include "includes.ihh"

namespace cuANN
{
__global__ void sum_columns ( 
                                float * input,
                                float * output, 
                                unsigned int w_size,
                                unsigned int i_size
                            );

}
#endif
