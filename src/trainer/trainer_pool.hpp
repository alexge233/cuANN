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
///
/// TODO: I can probably encapsulate the `worker_data` in this class, and then
///       find and pass to the `worker` object the actual worker data instance to use.
///
class trainer_pool
{   
public:

    /// Construct for @param max_threads
    trainer_pool (
                    unsigned int max_threads,
                    const std::vector<std::shared_ptr<trainer_data>> & thread_data
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
    void submit(cuANN::trainer & job);

private:

    unsigned int _max_threads_;
    boost::asio::io_service _io_service;
    boost::asio::io_service::work _work;
    std::vector<std::thread> _threads;
    std::condition_variable _cv;
    std::mutex _cvm;
    size_t _tasks = 0;
    const std::vector<std::shared_ptr<trainer_data>> & _thread_data;
};

}
#endif
