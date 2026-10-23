#include "paraforge/thread_pool.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>

using paraforge::ThreadPool;

TEST(ThreadPool, RejectsZeroWorkers) {
    EXPECT_THROW(ThreadPool(0), std::invalid_argument);
}

TEST(ThreadPool, RejectsEmptyTask) {
    ThreadPool pool(1);
    EXPECT_THROW(pool.submit(nullptr), std::invalid_argument);
}

TEST(ThreadPool, RunsEveryTaskBeforeWaitAllReturns) {
    ThreadPool pool(4);
    EXPECT_EQ(pool.size(), 4u);
    std::atomic<int> count{0};
    for (int i = 0; i < 1000; ++i) pool.submit([&] { count.fetch_add(1); });
    pool.wait_all();
    EXPECT_EQ(count.load(), 1000);
}

TEST(ThreadPool, WaitAllIsReusable) {
    ThreadPool pool(2);
    std::atomic<int> count{0};
    for (int round = 1; round <= 3; ++round) {
        for (int i = 0; i < 50; ++i) pool.submit([&] { count.fetch_add(1); });
        pool.wait_all();
        EXPECT_EQ(count.load(), round * 50);
    }
}

TEST(ThreadPool, WaitAllWithNoTasksReturns) {
    ThreadPool pool(2);
    pool.wait_all();
}

TEST(ThreadPool, WorkersRunConcurrently) {
    // Two tasks that each wait for the other can only both finish if two workers run at once.
    ThreadPool pool(2);
    std::atomic<int> arrived{0};
    std::atomic<int> met{0};
    auto rendezvous = [&] {
        arrived.fetch_add(1);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (arrived.load() < 2 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        if (arrived.load() == 2) met.fetch_add(1);
    };
    pool.submit(rendezvous);
    pool.submit(rendezvous);
    pool.wait_all();
    EXPECT_EQ(met.load(), 2);
}

TEST(ThreadPool, WaitAllRethrowsTaskExceptionAndPoolSurvives) {
    ThreadPool pool(2);
    std::atomic<int> count{0};
    pool.submit([] { throw std::runtime_error("boom"); });
    for (int i = 0; i < 10; ++i) pool.submit([&] { count.fetch_add(1); });
    EXPECT_THROW(pool.wait_all(), std::runtime_error);
    EXPECT_EQ(count.load(), 10);

    // Error is cleared after being reported; pool keeps working.
    pool.submit([&] { count.fetch_add(1); });
    pool.wait_all();
    EXPECT_EQ(count.load(), 11);
}

TEST(ThreadPool, DestructorDrainsQueuedTasks) {
    std::atomic<int> count{0};
    {
        ThreadPool pool(1);
        for (int i = 0; i < 100; ++i) pool.submit([&] { count.fetch_add(1); });
    }
    EXPECT_EQ(count.load(), 100);
}
