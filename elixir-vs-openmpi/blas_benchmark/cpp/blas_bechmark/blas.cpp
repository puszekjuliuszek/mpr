
#include <mpi.h>
#include <iostream>
#include <chrono>
#include <cblas.h>


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Wrong arguments, pass matrix size" << std::endl;
        return 0;
    }

    int N  = atoi(argv[1]);
    // int N = 10;

    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the total number of processes
    
    // const int SIZE = 2; // Message size
    // char message[SIZE] = "x"; // Message


    if (rank == 0) { // HEAD
        auto start_time = std::chrono::high_resolution_clock::now();
        double a[N * N];
        double b[N * N];
        double c[N * N];

        for (int i = 2; i < world_size; ++i) {
            MPI_Send(a, N * N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(b, N * N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }

        MPI_Recv(c, N * N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
              duration).count();

        std::cout << "For N = " << N << " Time (ms): " << milliseconds << std::endl;

    } else if (rank == 1) { // TAIL
        double c[N * N];
        for (int i = 2; i < world_size; ++i) {
            MPI_Recv(c, N * N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Send(c, N * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else { 
        double a[N * N];
        double b[N * N];
        double c[N * N];

        MPI_Recv(a, N * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(a, N * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            N,N,N, 1.0, a, 
            N, b, N, 0.0, 
            c, N);
        MPI_Send(c, N * N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}










