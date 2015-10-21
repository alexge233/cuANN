#include "sigmoid.hpp"
namespace cuANN
{

 __host__ __device__ float sigmoid ( const float x )
{
    return 1.0 / (1.0 + exp ( -x ) );
}

__host__ __device__ float fast_sigmoid ( const float x )
{
    return x / ( 1 + abs( x ) );
}

}
