// #include <mpi.h>
// #include <iostream>
// #include <vector>
// #include <iomanip>

// int main(int argc, char* argv[]) {
//     MPI_Init(&argc, &argv);
    
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (size < 2) {
//         if (rank == 0) std::cout << "This test requires at least 2 processes\n";
//         MPI_Finalize();
//         return 1;
//     }

//     const int ITERATIONS = 100;
//     std::vector<int> sizes = {8, 64, 512, 4096, 32768, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 1073741824, 2147483647};

//     if (rank == 0) {
//         std::cout << std::fixed << std::setprecision(2);
//         std::cout << "Size (bytes) | Avg RTT (ms) | Bandwidth (MB/s)\n";
//         std::cout << "-------------|--------------|-----------------\n";
        
//         for (int msg_size : sizes) {
//             std::vector<char> buffer(msg_size);
//             double start_time = MPI_Wtime();
            
//             for (int i = 0; i < ITERATIONS; i++) {
//                 MPI_Send(buffer.data(), msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
//                 MPI_Recv(buffer.data(), msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//             }
            
//             double total_time = MPI_Wtime() - start_time;
//             double avg_rtt = (total_time / ITERATIONS) * 1000; // milliseconds
            
//             // Cast msg_size to double to prevent integer overflow
//             double bytes_transferred = static_cast<double>(msg_size) * ITERATIONS * 2.0;
//             double bandwidth = (bytes_transferred / total_time) / 1048576.0; // Bytes/s to MB/s (using 2^20)
            
//             std::cout << std::setw(12) << msg_size << " | "
//                       << std::setw(12) << avg_rtt << " | "
//                       << std::setw(15) << bandwidth << "\n";
//         }
//     }
//     else if (rank == 1) {
//         for (int msg_size : sizes) {
//             std::vector<char> buffer(msg_size);
//             for (int i = 0; i < ITERATIONS; i++) {
//                 MPI_Recv(buffer.data(), msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//                 MPI_Send(buffer.data(), msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
//             }
//         }
//     }

//     MPI_Finalize();
//     return 0;
// }

#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) std::cout << "This test requires at least 2 processes\n";
        MPI_Finalize();
        return 1;
    }

    const int ITERATIONS = 100;
    std::vector<int> sizes = {8, 64, 512, 4096, 32768, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 1073741824, 2147483647};

    // Open CSV file only on rank 0
    std::ofstream csv_file;
    if (rank == 0) {        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Size (bytes) | Avg RTT (ms) | Bandwidth (MB/s)\n";
        std::cout << "-------------|--------------|-----------------\n";
        
        int msg_size = 1;
        for (int jjj = 1; jjj <= 100; jjj++) {
            msg_size *= 8 ;
            csv_file.open("mpi_results.csv", std::ios::app);

            std::vector<char> buffer(msg_size);
            double start_time = MPI_Wtime();
            
            for (int i = 0; i < ITERATIONS; i++) {
                MPI_Send(buffer.data(), msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer.data(), msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            double total_time = MPI_Wtime() - start_time;
            double avg_rtt = (total_time / ITERATIONS) * 1000; // milliseconds
            
            double bytes_transferred = static_cast<double>(msg_size) * ITERATIONS * 2.0;
            double bandwidth = (bytes_transferred / total_time) / 1048576.0; // Bytes/s to MB/s
            
            // Write to console
            std::cout << std::setw(12) << msg_size << " | "
                      << std::setw(12) << avg_rtt << " | "
                      << std::setw(15) << bandwidth << "\n";
            
            // Write to CSV file
            csv_file << msg_size << "," << avg_rtt << "," << bandwidth << "\n";
            csv_file.close();
        }
        
    }
    else if (rank == 1) {
        int msg_size = 1;
        for (int jjj = 1; jjj <= 100; jjj++) {
            msg_size *= 8;
            std::vector<char> buffer(msg_size);
            for (int i = 0; i < ITERATIONS; i++) {
                MPI_Recv(buffer.data(), msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer.data(), msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}