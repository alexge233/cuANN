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
                 )
    : _max_threads_(max_threads), _work(_io_service), _thread_data(thread_data)
    {}

    /// Start processing threads
    void start()
    {
        _threads.reserve(_max_threads_);
        for (int i = 0 ; i < _max_threads_; ++i)
            _threads.emplace_back(std::bind(&trainer_pool::thread_proc, this));
    }
    
    /// Wait for the next available slot
    void wait()
    {   
        std::unique_lock<std::mutex> lock(_cvm); 
        _cv.wait(lock, [this] { return _tasks == 0; });
    }
    
    /// Wait for all threads to finish and stop
    void stop()
    {   
        wait();
        _io_service.stop();
        for (auto& t : _threads) 
        {
            if (t.joinable())
                t.join();
        }
        _threads.clear();
    }

    /// Process threads 
    void thread_proc()
    {   
        while (!_io_service.stopped())
            _io_service.run();
    }
    
    /// Reduce count
    void reduce() 
    {
        std::unique_lock<std::mutex> lock(_cvm);
        if (--_tasks == 0) {
            lock.unlock();
            _cv.notify_all();
        }
    }
    
    /// Submit a new Job (worker)    
    template <class A,class D>
    void submit(cuANN::trainer<A,D> & job)
    {
        std::unique_lock<std::mutex> lock(_cvm);
        ++ _tasks;
        lock.unlock();
        _io_service.post([this,job] () mutable
                         {
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
