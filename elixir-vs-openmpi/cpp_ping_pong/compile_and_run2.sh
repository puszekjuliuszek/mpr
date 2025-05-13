#!/bin/bash
#SBATCH -J plgmpr25-cpu
#SBATCH -o plgmpr25-cpu-%a-%j.out
#SBATCH -e plgmpr25-cpu-%a-%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=6G
#SBATCH -p plgrid
#SBATCH --time=00:05:00



# echo "Running demo"
# mpicxx -std=c++11 mpiDemo.cpp -o compiled/mpiDemo
# mpirun -np 4 ./compiled/mpiDemo

# echo "Running ping pong benchmark"
# mpicxx -std=c++11 pingPong1.cpp -o compiled/pingPong1
# mpirun -np 4 ./compiled/pingPong1

module load openmpi

echo "Running ping pong benchmark 2 "
mpicxx -std=c++11 pingPong2.cpp -o compiled/pingPong2
mpirun -n 2 ./compiled/pingPong2

# echo "Running ping pong benchmark 3 "
# mpicxx -std=c++11 pingPong3.cpp -o compiled/pingPong3
# mpirun -np 4 ./compiled/pingPong3

# echo "Running all to all benchmark"
# mpicxx -std=c++11 allToAll.cpp -o compiled/allToAll
# mpirun -np 4 ./compiled/allToAll

# echo "Running ring benchmark"
# mpicxx -std=c++11 ring.cpp -o compiled/ring
# mpirun -np 4 ./compiled/ring