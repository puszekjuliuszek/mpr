#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <array>
#include <limits>
#include <cstdint>

void run_test(int64_t iterations, int message_size, int rank, std::ofstream& csv_file) {
    std::vector<char> buffer(message_size);
    long double start_time, end_time, total_time;

    if (rank == 0) {
        // Ensure buffer is initialized to prevent potential optimization issues
        std::fill(buffer.begin(), buffer.end(), 0);
        
        start_time = MPI_Wtime();
        
        for (int64_t i = 0; i < iterations; i++) {
            MPI_Send(buffer.data(), message_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(buffer.data(), message_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        end_time = MPI_Wtime();
        total_time = end_time - start_time;

        // Use long double for maximum precision
        long double iterations_ld = static_cast<long double>(iterations);
        long double bytes_per_msg = static_cast<long double>(message_size);
        long double total_bytes = bytes_per_msg * iterations_ld * 2.0L; // *2 for send+receive
        
        // Check for negative or unreasonable results
        if (total_time <= 0.0) {
            std::cout << "Error: Timing resolution too low for accurate measurement\n";
            return;
        }

        long double avg_rtt = (total_time / iterations_ld) * 1000.0L; // ms
        long double msg_rate = iterations_ld / total_time; // messages/s
        long double bandwidth = total_bytes / (total_time * 1000000.0L); // MB/s

        // Verify results make sense
        if (bandwidth < 0 || avg_rtt < 0) {
            std::cout << "Error: Negative values detected - possible overflow\n";
            return;
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\nIterations: " << iterations << "\n";
        std::cout << "Message Size: " << message_size << " bytes\n";
        std::cout << "Avg Round-Trip Time: " << avg_rtt << " ms\n";
        std::cout << "Message Rate: " << msg_rate << " messages/s\n";
        std::cout << "Bandwidth: " << bandwidth << " MB/s\n";

            csv_file.open("mpi_pingpong_results.csv", std::ios_base::app);
    
        csv_file << std::setprecision(10) << iterations << "," 
                 << message_size << "," 
                 << avg_rtt << "," 
                 << msg_rate << "," 
                 << bandwidth << "\n";
    
                 csv_file.close();

    }
    else if (rank == 1) {
        for (int64_t i = 0; i < iterations; i++) {
            MPI_Recv(buffer.data(), message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buffer.data(), message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            std::cout << "This test requires at least 2 processes\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int MESSAGE_SIZE = 1024; // 1KB
    // const int64_t MAX_ITERATIONS = 100000000; // Match your test case

    std::ofstream csv_file;
    if (rank == 0) {
        csv_file.open("mpi_pingpong_results.csv", std::ios_base::app);
        csv_file << "Iterations,MessageSize(bytes),AvgRTT(ms),MessageRate(msg/s),Bandwidth(MB/s)\n";
        csv_file.close();
    }

    int64_t iterations = 5;
    for (int i = 0; i < 100; i++) { // Up to 10^7
        MPI_Barrier(MPI_COMM_WORLD);
        
        // if (iterations > MAX_ITERATIONS) {
            // if (rank == 0) {
                // std::cout << "Stopping at " << iterations << " iterations\n";
            // }
            // break;
        // }
        iterations *= 2;
        run_test(iterations, MESSAGE_SIZE, rank, csv_file);
    }

    if (rank == 0) {
        csv_file.close();
        std::cout << "\nResults saved to mpi_pingpong_results.csv\n";
    }

    MPI_Finalize();
    return 0;
}