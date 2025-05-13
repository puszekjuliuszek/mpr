
# build
module load openblas
mpicxx -std=c++11 blas.cpp -o build/blas -lopenblas


# N = 100
mpirun -np 6 ./build/blas 500

# N = 150
mpirun -np 6 ./build/blas 1000

# N = 200
mpirun -np 6 ./build/blas 1500

# N = 250
mpirun -np 6 ./build/blas 2000
