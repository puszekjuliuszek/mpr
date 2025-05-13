#include <mpi.h>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    const int num_iterations = 100000;
    char message[2] = "x";  // Message size (we're using a message of size 1 character)
    char recv_message[2];
    
    // Synchronize processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        // Send message to the next process and receive from the previous one in the ring
        MPI_Sendrecv(message, sizeof(message), MPI_CHAR,
                     (rank + 1) % size, 0,   // Send to next
                     recv_message, sizeof(recv_message), MPI_CHAR,
                     (rank + size - 1) % size, 0, // Receive from previous
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes have finished

    std::chrono::duration<double> duration = end_time - start_time;
    
    if (rank == 0) {
        std::cout << "Time for " << num_iterations << " iterations with " 
                  << size << " processes is " << duration.count() << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}