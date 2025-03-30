#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>
#include <thread>
#include <mutex>

std::mutex cout_mutex; // Mutex for synchronized output

void monte_carlo_simulation(long points_per_thread, long& count_inside_circle) {
    long local_count = 0;
    for (long i = 0; i < points_per_thread; ++i) {
        double x = static_cast<double>(std::rand()) / RAND_MAX;
        double y = static_cast<double>(std::rand()) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }
    count_inside_circle += local_count;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <number_of_points> <number_of_runs> <number_of_threads>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    long total_points = std::stol(argv[1]);
    int num_runs = std::stoi(argv[2]);
    int num_threads = std::stoi(argv[3]);
    long points_per_process = total_points / size;
    long points_per_thread = points_per_process / num_threads;
    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr) + rank)); // Different seed for each process

    auto total_time = 0.0;

    for (int run = 0; run < num_runs; ++run) {
        long count_inside_circle = 0;

        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();

        // Create threads for the Monte Carlo simulation
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(monte_carlo_simulation, points_per_thread, std::ref(count_inside_circle));
        }

        // Wait for all threads to complete
        for (auto& th : threads) {
            th.join();
        }

        // Timing end
        auto end_time = std::chrono::high_resolution_clock::now();

        // Reduce all counts to the rank 0 process
        long total_inside_circle;
        MPI_Reduce(&count_inside_circle, &total_inside_circle, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        // Calculate execution time
        std::chrono::duration<double> duration = end_time - start_time;
        total_time += duration.count();

        // Output result on rank 0 process
        if (rank == 0) {
            double pi_estimate = 4.0 * total_inside_circle / total_points;
            std::lock_guard<std::mutex> guard(cout_mutex); // Ensure synchronized output
            // std::cout << "Run " << (run + 1) << ": Estimated value of pi with "
            //           << total_points << " points: " << pi_estimate 
            //           << " (Execution time: " << duration.count() << " seconds)" << std::endl;
        }
    }

    // Output the total time on rank 0
    if (rank == 0) {
        std::lock_guard<std::mutex> guard(cout_mutex); // Ensure synchronized output
        std::cout << "Total execution time for " << num_runs << " runs: " 
                  << total_time << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}