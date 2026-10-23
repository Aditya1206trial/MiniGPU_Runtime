#include "paraforge/thread_pool.hpp"

#include <stdexcept>
#include <utility>

namespace paraforge {

ThreadPool::ThreadPool(std::size_t num_workers) {
    if (num_workers == 0) {
        throw std::invalid_argument("ThreadPool: num_workers must be > 0");
    }
    workers_.reserve(num_workers);
    try {
        for (std::size_t i = 0; i < num_workers; ++i) {
            workers_.emplace_back(&ThreadPool::worker_loop, this);
        }
    } catch (...) {
        // Destructor won't run; joinable threads left in workers_ would std::terminate.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        task_ready_.notify_all();
        for (auto& w : workers_) {
            w.join();
        }
        throw;
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    task_ready_.notify_all();
    for (auto& w : workers_) {
        w.join();
    }
}

void ThreadPool::submit(std::function<void()> task) {
    if (!task) {
        throw std::invalid_argument("ThreadPool::submit: empty task");
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push(std::move(task));
    }
    task_ready_.notify_one();
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(mutex_);
    all_done_.wait(lock, [this] { return tasks_.empty() && active_ == 0; });
    if (first_error_) {
        std::exception_ptr err = std::exchange(first_error_, nullptr);
        std::rethrow_exception(err);
    }
}

void ThreadPool::worker_loop() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            task_ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
            if (tasks_.empty()) {
                return;  // stopping and drained
            }
            task = std::move(tasks_.front());
            tasks_.pop();
            ++active_;
        }
        std::exception_ptr err;
        try {
            task();
        } catch (...) {
            err = std::current_exception();  // an escaping exception would std::terminate the worker
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (err && !first_error_) {
                first_error_ = err;
            }
            --active_;
            if (tasks_.empty() && active_ == 0) {
                all_done_.notify_all();
            }
        }
    }
}

}  // namespace paraforge
