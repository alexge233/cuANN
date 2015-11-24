#include "trainer_pool.hpp"

namespace cuANN
{

trainer_pool::trainer_pool ( 
                             unsigned int max_threads,
                             const std::vector<std::shared_ptr<trainer_data>> & thread_data
                           )
: _max_threads_(max_threads), _work(_io_service), _thread_data(thread_data)
{}
 
void trainer_pool::start()
{
    _threads.reserve(_max_threads_);
    for (int i = 0 ; i < _max_threads_; ++i)
        _threads.emplace_back(std::bind(&trainer_pool::thread_proc, this));
}
    
void trainer_pool::wait()
{   
    std::unique_lock<std::mutex> lock(_cvm); 
    _cv.wait(lock, [this] { return _tasks == 0; });
}
    
void trainer_pool::stop()
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
   
void trainer_pool::thread_proc()
{   
    while (!_io_service.stopped())
        _io_service.run();
}

void trainer_pool::reduce() 
{
    std::unique_lock<std::mutex> lock(_cvm);
    if (--_tasks == 0) {
        lock.unlock();
        _cv.notify_all();
    }
}

void trainer_pool::submit(cuANN::trainer & job)
{
    // TODO: Search into trainer_data vector and find the first `available` object

    std::unique_lock<std::mutex> lock(_cvm);
    ++ _tasks;
    lock.unlock();
    _io_service.post([this,job] () mutable
                     {
                         job();
                         reduce();
                     });
}
}
