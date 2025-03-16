#include <mpi.h>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    const int SIZE = 2; // Message size
    char message[SIZE] = "x"; // Message
    int num_iterations = 100000; // Number of iterations

    if (rank == 0) {
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            MPI_Send(message, SIZE, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(message, SIZE, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        
        std::cout << "Rank 0: Time for " << num_iterations << " iterations is " 
                  << duration.count() << " seconds." << std::endl;
    } else if (rank == 1) {
        for (int i = 0; i < num_iterations; ++i) {
            MPI_Recv(message, SIZE, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(message, SIZE, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}