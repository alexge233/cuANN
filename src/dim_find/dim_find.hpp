#ifndef _cuANN_dim_find_HPP_
#define _cuANN_dim_find_HPP_
#include "includes.ihh"
namespace cuANN
{
/// CUDA Block X,Y for 2D Grid: 1024 threads per block
/// Max number of threads per block for 2D grid = BLOCK_X * BLOCK_Y
/// The Max depends upon CUDA Compute (3.0 in this case) and the GPU Device used
#define BLOCK_X 32
#define BLOCK_Y 32 

/// Find dimensions for grid, blocks and threads using 2D
/// This is used to propagate vector to matrix vector (multiplication of vectors)
struct dim_find_2d
{
    unsigned int num_blocks_x;
    unsigned int num_blocks_y;
    unsigned int thread_blocks_x;
    unsigned int thread_blocks_y;

    
    dim_find_2d ( unsigned int x_size, unsigned int y_size )
    {
        if ( x_size < BLOCK_X )
        {
            num_blocks_x = 1;
            thread_blocks_x = x_size;
        }
        else
        {
            num_blocks_x = ( x_size / BLOCK_X ) +1;
            thread_blocks_x = BLOCK_X;
        }
        
        if ( y_size < BLOCK_Y )
        {
            num_blocks_y = 1;
            thread_blocks_y = y_size;
        }
        else
        {
            num_blocks_y = ( y_size / BLOCK_Y ) +1;
            thread_blocks_y = BLOCK_Y;
        }
    }
};

}
#endif
