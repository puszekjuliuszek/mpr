#!/bin/bash
echo "Running demo"
mpicxx -std=c++11 mpiDemo.cpp -o compiled/mpiDemo
mpirun -np 4 ./compiled/mpiDemo

echo "Running ping pong benchmark"
mpicxx -std=c++11 pingPong.cpp -o compiled/pingPong
mpirun -np 4 ./compiled/pingPong

echo "Running all to all benchmark"
mpicxx -std=c++11 allToAll.cpp -o compiled/allToAll
mpirun -np 4 ./compiled/allToAll

echo "Running ring benchmark"
mpicxx -std=c++11 ring.cpp -o compiled/ring
mpirun -np 4 ./compiled/ring