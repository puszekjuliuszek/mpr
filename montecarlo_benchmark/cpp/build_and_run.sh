#!/bin/bash

# Build the executable
mkdir -p build
mpicxx -std=c++11 montecarlo.cpp -o build/montecarlo -lpthread

# Iterate through a range of points
# You can adjust the starting and ending points as needed.
for points in  100000000; do
    echo "Running with $points points"

    # Iterate through a range of threads
    for threads in 70 80 90 100; do
        echo "Running with $threads threads"
        mpirun -np 4 ./build/montecarlo $points 20 $threads
    done
done