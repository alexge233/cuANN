#ifndef _cuANN_trainer_pool_HPP_
#define _cuANN_trainer_pool_HPP_
#include "includes.ihh"
#include "trainer.hpp"
namespace cuANN 
{
/// @author Alexander Giokas <a.gkiokas@warwick.ac.uk>
/// @date   November 2015
/// @version 1
/// @brief Trainer thread pool, used to control parallel execution of training threads.
class trainer_pool
{   
public:

    /// Construct for @param max_threads
    trainer_pool ( 
                    unsigned int max_threads,
                    std::vector<std::shared_ptr<trainer_data>> & thread_data
                 );

    /// Start processing threads
    void start();
    
    /// Wait for the next available slot
    void wait();
    
    /// Wait for all threads to finish and stop
    void stop();
    
    /// Process threads 
    void thread_proc();
    
    /// Reduce count
    void reduce();
    
    /// Submit a new Job (worker)    
    template <class A,class D>
    void submit( cuANN::trainer<A,D> & job)
    {
        std::unique_lock<std::mutex> lock(_cvm);
        ++ _tasks;
        lock.unlock();
        _io_service.post([this,job] () mutable
                         {
                             // Need to pass `_thread_data` as a parameter here
                             job(_thread_data);
                             reduce();
                         });
    }

private:

    unsigned int _max_threads_;
    boost::asio::io_service _io_service;
    boost::asio::io_service::work _work;
    std::vector<std::thread> _threads;
    std::vector<std::shared_ptr<trainer_data>> & _thread_data;
    std::condition_variable _cv;
    std::mutex _cvm;
    size_t _tasks = 0;
};

}
#endif
