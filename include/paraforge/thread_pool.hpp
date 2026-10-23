#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace paraforge {

// Fixed number of workers pulling tasks from one mutex-guarded FIFO queue.
class ThreadPool {
public:
    // Throws std::invalid_argument if num_workers == 0.
    explicit ThreadPool(std::size_t num_workers);
    // Runs every already-queued task, then joins workers.
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void submit(std::function<void()> task);

    // Blocks until the queue is empty and no task is running.
    // Rethrows the first exception thrown by a task since the last wait_all.
    // Must not be called from inside a task (it would wait on itself).
    void wait_all();

    std::size_t size() const { return workers_.size(); }

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable task_ready_;
    std::condition_variable all_done_;
    std::size_t active_ = 0;
    bool stopping_ = false;
    std::exception_ptr first_error_;
};

}  // namespace paraforge
