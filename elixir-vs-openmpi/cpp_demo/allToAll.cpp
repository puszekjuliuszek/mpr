#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int message_size = 10; // Each message will contain 10 characters
    std::vector<char> send_buffer(size * message_size);   // Sending buffer
    std::vector<char> recv_buffer(size * message_size);    // Receiving buffer

    // Fill the send buffer with a unique message for each process
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < message_size; ++j) {
            send_buffer[i * message_size + j] = 'A' + rank; // Fill the buffer with rank character
        }
    }

    // Synchronize processes
    MPI_Barrier(MPI_COMM_WORLD);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Use MPI_Alltoall instead of manually sending and receiving
    MPI_Alltoall(send_buffer.data(), message_size, MPI_CHAR, 
                  recv_buffer.data(), message_size, MPI_CHAR, 
                  MPI_COMM_WORLD);

    auto end_time = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes have finished

    std::chrono::duration<double> duration = end_time - start_time;

    if (rank == 0) {
        std::cout << "All-to-All Communication time with " 
                  << size << " processes is " << duration.count() << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}