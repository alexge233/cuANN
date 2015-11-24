#include "trainer.hpp"

namespace cuANN
{

trainer::trainer (
                   const std::shared_ptr<cuANN::trainer_data> gpu_data,
                   const thrust::host_vector<float> & input,
                   const thrust::host_vector<float> & output,
                   float alpha,
                   float epsilon,
                   unsigned int index
                )
: _dmem(gpu_data), _a(alpha), _e(epsilon), _i(index)
{
    assert(gpu_data && _dmem);
    // TODO: WARNING!!! Apart from running a training Iteration
    //                  1) FIND an available `trainer_data` object to use or throw exception
    //                  2) CONTROL the availability of the `trainer_data` object being used
    //                  
    //                  Also NOTE: 
    //                  1) We may have to RESET (ZERO-IN) SOME of the data of `trainer_data`
    //                  2) We have to COPY 
}

__host__ void trainer::operator()() const
{
    // TODO: REMEMBER To RELEASE (MARK AS AVAILABLE) the `trainer_data` object once we've finished!!!
}

__host__ void trainer::forward_prop()
{

}

__host__ void trainer::delta_output()
{

}

__host__ void trainer::delta_hidden()
{

}

__host__ void trainer::grad_calc()
{

}

__host__ void trainer::error_calc()
{

}

};
