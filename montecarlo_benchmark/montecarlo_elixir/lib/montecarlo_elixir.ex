defmodule MonteCarlo do
  def estimate_pi(points_per_thread) do
    1..points_per_thread
    |> Enum.reduce(0, fn _, acc ->
      x = :rand.uniform()
      y = :rand.uniform()

      if x * x + y * y <= 1.0 do
        acc + 1
      else
        acc
      end
    end)
  end

  def run(total_points, num_runs, num_threads) do
    points_per_thread = div(total_points, num_threads)

    total_time =
      for run <- 1..num_runs do
        # Start timing
        start_time = System.monotonic_time()

        # Create tasks for each thread
        tasks = for _ <- 1..num_threads do
          Task.async(fn -> estimate_pi(points_per_thread) end)
        end

        # Await results and sum them up
        count_inside_circle = tasks |> Enum.map(&Task.await/1) |> Enum.sum()

        # Timing end
        end_time = System.monotonic_time()
        
        # Calculate execution time
        duration_ms = System.convert_time_unit(end_time - start_time, :native, :millisecond)

        pi_estimate = 4.0 * count_inside_circle / total_points


        # Return the duration for summation if needed
        duration_ms
      end
      |> Enum.sum()

    total_time_seconds = total_time / 1000.0
    IO.puts "Total execution time for #{num_runs} runs: #{total_time} ms, or #{total_time_seconds} seconds"
  end
end

# Run the application from the command line
total_points = String.to_integer(System.argv() |> Enum.at(0, "10000000"))
num_runs = String.to_integer(System.argv() |> Enum.at(1, "1"))
num_threads = String.to_integer(System.argv() |> Enum.at(2, "1"))
MonteCarlo.run(total_points, num_runs, num_threads)