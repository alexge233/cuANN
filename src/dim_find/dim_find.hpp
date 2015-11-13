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

/// CUDA BLOCK 1 Dimension, using max 1024 threads per block
#define BLOCK_1D 1024

/// Find dimensions for grid, blocks and threads using 2D
/// This is used to propagate vector to matrix vector (multiplication of vectors)
struct dim2D
{
    unsigned int num_blocks_x;
    unsigned int num_blocks_y;
    unsigned int block_threads_x;
    unsigned int block_threads_y;

    
    dim2D ( unsigned int x_size, unsigned int y_size )
    {
        if ( x_size < BLOCK_X )
        {
            num_blocks_x = 1;
            block_threads_x = x_size;
        }
        else
        {
            num_blocks_x = ( x_size / BLOCK_X ) +1;
            block_threads_x = BLOCK_X;
        }
        
        if ( y_size < BLOCK_Y )
        {
            num_blocks_y = 1;
            block_threads_y = y_size;
        }
        else
        {
            num_blocks_y = ( y_size / BLOCK_Y ) +1;
            block_threads_y = BLOCK_Y;
        }
    }
};

/// Find dimensions for 1D grid
struct dim1D
{
    unsigned int num_blocks_x;
    unsigned int block_threads_x;


    dim1D ( unsigned int count )
    {
        if ( count < BLOCK_1D )
        {
            num_blocks_x = 1;
            block_threads_x = count;
        }
        else
        {
            num_blocks_x = ( count / BLOCK_1D ) +1;
            block_threads_x = BLOCK_1D;
        }
    }
};

}
#endif
