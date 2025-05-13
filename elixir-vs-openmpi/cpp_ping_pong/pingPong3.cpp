#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2 || size % 2 != 0) {
        if (rank == 0) {
            std::cout << "This test requires an even number of processes (>=2)\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int ITERATIONS = 1000;
    const int MESSAGE_SIZE = 1024; // 1KB
    std::vector<char> buffer(MESSAGE_SIZE);
    double start_time, end_time, total_time;

    // Pair processes: 0-1, 2-3, 4-5, etc.
    int pair_partner = (rank % 2 == 0) ? rank + 1 : rank - 1;
    bool is_sender = (rank % 2 == 0);

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
    start_time = MPI_Wtime();

    if (is_sender) {
        for (int i = 0; i < ITERATIONS; i++) {
            MPI_Send(buffer.data(), MESSAGE_SIZE, MPI_CHAR, pair_partner, 0, MPI_COMM_WORLD);
            MPI_Recv(buffer.data(), MESSAGE_SIZE, MPI_CHAR, pair_partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else {
        for (int i = 0; i < ITERATIONS; i++) {
            MPI_Recv(buffer.data(), MESSAGE_SIZE, MPI_CHAR, pair_partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buffer.data(), MESSAGE_SIZE, MPI_CHAR, pair_partner, 0, MPI_COMM_WORLD);
        }
    }

    end_time = MPI_Wtime();
    total_time = end_time - start_time;

    // Gather results from all processes
    std::vector<double> all_times(size);
    MPI_Gather(&total_time, 1, MPI_DOUBLE, all_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double avg_time = 0;
        for (int i = 0; i < size; i++) {
            avg_time += all_times[i];
        }
        avg_time /= size;

        double avg_rtt = (avg_time / ITERATIONS) * 1e6; // microseconds
        double msg_rate = ITERATIONS / avg_time; // messages per second
        double bandwidth = (MESSAGE_SIZE * ITERATIONS * 2.0) / (avg_time * 1e6); // MB/s

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Scalability Test Results:\n";
        std::cout << "Number of process pairs: " << size/2 << "\n";
        std::cout << "Message Size: " << MESSAGE_SIZE << " bytes\n";
        std::cout << "Iterations per pair: " << ITERATIONS << "\n";
        std::cout << "Avg Round-Trip Time: " << avg_rtt << " μs\n";
        std::cout << "Message Rate: " << msg_rate << " messages/s\n";
        std::cout << "Bandwidth: " << bandwidth << " MB/s\n";
    }

    MPI_Finalize();
    return 0;
}