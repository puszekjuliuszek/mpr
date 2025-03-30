# Iterate through a range of points
# You can adjust the starting and ending points as needed.
for points in 10000000 20000000 30000000 40000000 50000000 60000000 70000000 80000000 90000000 100000000; do
    echo "Running with $points points"

    # Iterate through a range of threads
    for threads in 1 10 20 30 40 50 60 70 80 90 100; do
        mix clean
        echo "Running with $threads threads"
        mix run --no-halt -- $points 20 $threads
    done
done