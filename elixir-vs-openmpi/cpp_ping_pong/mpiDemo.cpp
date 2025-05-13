#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Initialize the MPI environment

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the total number of processes

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the current process

    const int master_rank = 0; // Define the rank of the master process
    if (world_rank == master_rank) {
        // Master process
        const int message_count = world_size - 1; // Number of workers
        std::vector<int> results(message_count);

        // Send a message to each worker
        for (int i = 1; i < world_size; ++i) {
            int message = i; // Sample message (rank number)
            MPI_Send(&message, 1, MPI_INT, i, 0, MPI_COMM_WORLD); // Send message
        }

        // Receive results from each worker
        for (int i = 1; i < world_size; ++i) {
            MPI_Recv(&results[i - 1], 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Master received result from worker " << i << ": " << results[i - 1] << std::endl;
        }
    } else {
        // Worker process
        int number;
        MPI_Recv(&number, 1, MPI_INT, master_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive message
        int result = number * number; // Example processing: square the number

        MPI_Send(&result, 1, MPI_INT, master_rank, 0, MPI_COMM_WORLD); // Send result back to master
        std::cout << "Worker " << world_rank << " processed message: " << number << " -> result: " << result << std::endl;
    }

    MPI_Finalize(); // Clean up and finalize MPI environment
    return 0;
}