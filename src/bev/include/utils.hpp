#include <iostream>
#include <chrono>
#include <string>

// #define ENABLE_TIMING

class Timer {
public:
    // Constructor starts the timer and stores the message
    Timer(const std::string& message)
        : start_time_(std::chrono::high_resolution_clock::now()), message_(message) {
    }

    // Destructor stops the timer and calculates elapsed time
    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
        std::cout << "Timer stopped: " << message_ << " | Duration: " << duration.count() << " ms" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    std::string message_;
};
