defmodule ElixirMPI.Benchmark do
  @moduledoc """
  Benchmark module for ElixirMPI to measure inter-process communication performance.
  """

  @doc """
  Runs a ping-pong benchmark between two processes.
  Measures the time it takes to send a message back and forth between processes.

  ## Parameters
  - iterations: Number of ping-pong cycles to execute
  - message_size: Size of the test message in bytes
  """
  def ping_pong(iterations \\ 1000, message_size \\ 1024) do
    IO.puts("Starting ping-pong benchmark")
    IO.puts("Iterations: #{iterations}, Message size: #{message_size} bytes")

    # Initialize MPI with 2 processes for ping-pong
    {rank, size, world} = ElixirMPI.MPI.init(2, fn rank, size, world ->
      ping_pong_process(rank, size, world, iterations, message_size)
    end)

    IO.puts("Main process: Initialized MPI with rank=#{rank}, size=#{size}")

    # Wait for processes to complete
    ElixirMPI.MPI.finalize()

    IO.puts("Ping-pong benchmark completed")
  end

  @doc """
  The function executed by each process in the ping-pong benchmark.
  Process 0 is the ping process, Process 1 is the pong process.
  """
  def ping_pong_process(rank, size, world, iterationsttt, message_size) do
    # Create test data of specified size
    test_data = generate_test_data(message_size)
    max_repeats=100

    case rank do
      0 ->
        # Process 0 (ping)
        IO.puts("Process #{rank}: Starting ping process")

        for r<-1..max_repeats do
        # Measure total time for all iterations
        start_time = :os.system_time(:micro_seconds)

        iterations=round(10 * :math.pow(2, r-1))

        # Run ping-pong for specified iterations
        Enum.each(1..iterations, fn i ->
          # Send ping
          ElixirMPI.MPI.send(test_data, 1, i, world)

          # Receive pong
          {:ok, _received_data, _from_pid, _tag} = ElixirMPI.MPI.recv(1, i, 5000)


        end)

        end_time = :os.system_time(:micro_seconds)
        total_time = end_time - start_time

        # Calculate performance metrics
        total_messages = iterations * 2  # Each iteration has 2 messages (ping + pong)
        total_data = total_messages * message_size

        # Calculate average round-trip time
        avg_rtt = total_time / total_messages / 1_000  # Convert to milliseconds

        # Calculate messages per second
        msgs_per_sec = total_messages / (total_time / 1_000_000)  # Convert to seconds

        # Calculate bandwidth (bytes per second)
        total_seconds = total_time / 1_000_000
        bandwidth = (total_data / total_seconds) / 1_048_576  # MB/s

        {:ok, file} = File.open("pingping_results_#{rank}.txt", [:append, :utf8])

        avg_rtt_str = :erlang.float_to_binary(avg_rtt, [:compact, decimals: 20])
        msgs_per_sec_str = :erlang.float_to_binary(Float.round(msgs_per_sec, 2), [:compact, decimals: 20])

        write_to_file(file, "#{iterations};#{avg_rtt_str};#{msgs_per_sec_str}")

        # Close the file
        File.close(file)



        # # Report results
        # IO.puts("\n==== Ping-Pong Benchmark Results ====")
        # IO.puts("Total time: #{total_time} ms")
        # IO.puts("Message size: #{message_size} bytes")
        # IO.puts("Iterations: #{iterations}")
        # IO.puts("Total messages: #{total_messages}")
        # IO.puts("Average round-trip time: #{Float.round(avg_rtt, 3)} ms")
        # IO.puts("Message rate: #{Float.round(msgs_per_sec, 2)} messages/second")
        # IO.puts("Bandwidth: #{Float.round(bandwidth, 2)} KB/s")
        # IO.puts("=====================================\n")

      end
      1 ->
        # Process 1 (pong)
        IO.puts("Process #{rank}: Starting pong process")

        for r<-1..max_repeats do

          iterations = round(10 * :math.pow(2, r-1))

        Enum.each(1..iterations, fn i ->
          # Receive ping using direct PID lookup from world map
          {:ok, received_data, from_pid, _tag} = ElixirMPI.MPI.recv(Map.get(world, 0), i, 5000)

          # Send pong (echo back)
          ElixirMPI.MPI.send(received_data, 0, i, world)
        end)

      end
    end
  end

  @doc """
  Generates test data of specified size in bytes.
  """
  def generate_test_data(size) do
    # Generate a binary of the specified size
    initial_data = :crypto.strong_rand_bytes(8)
    :binary.copy(initial_data, size)
  end

  @doc """
  Runs a latency vs. message size benchmark to measure how latency changes
  with different message sizes.

  ## Parameters
  - sizes: List of message sizes to test
  - iterations: Number of ping-pong iterations for each size
  """
  def latency_vs_size(sizes \\ [1, 512, 1024, 4096, 16384, 65536], iterations \\ 100) do
    IO.puts("Starting latency vs. message size benchmark")

    # Initialize MPI with 2 processes
    {rank, size, world} = ElixirMPI.MPI.init(2, fn rank, size, world ->
      latency_size_process(rank, size, world, sizes, iterations)
    end)

    IO.puts("Main process: Initialized MPI with rank=#{rank}, size=#{size}")

    # Wait for processes to complete
    ElixirMPI.MPI.finalize()

    IO.puts("Latency vs. size benchmark completed")
  end

  @doc """
  The function executed by each process in the latency vs. size benchmark.
  """
  def latency_size_process(rank, size, world, sizes, iterations) do
    # Ensure all processes are ready
    ElixirMPI.MPI.barrier(rank)

    # Generate dynamic sizes (starting at 8, multiplying by 8 each time)
    sizes_count = 100

    # Open file for writing (will be appended to)

    case rank do
      0 ->
        test_data = "dddddddd"

        for i <- 1..sizes_count do
          msg_size = round(8 * :math.pow(8, i-1))

          test_data = Enum.reduce(1..(div(msg_size, 8)), test_data, fn _, acc -> acc <> "dddddddd" end)

          IO.puts("Testing message size: #{byte_size(test_data)} bytes")

          # Notify process 1 which size we're testing
          ElixirMPI.MPI.send(msg_size, 1, 0, world)

          # Measure time for all iterations
          start_time = :os.system_time(:micro_seconds)

          test = 0

          Enum.each(1..iterations, fn i ->
            # Send ping with size-specific tag
            ElixirMPI.MPI.send(test_data, 1, msg_size, world)

            # Receive pong - use the world map to get the PID directly
            receiver_pid = Map.get(world, 1)
            {:ok, _received_data, _from_pid, _tag} = ElixirMPI.MPI.recv(receiver_pid, msg_size, 50000000)
          end)

          end_time = :os.system_time(:micro_seconds)
          total_time = end_time - start_time

          # Calculate average round trip time in milliseconds
          avg_rtt = total_time / iterations / 1000

          # Calculate bandwidth (bytes per second in MB/s)
          total_bytes = msg_size * 2 * iterations
          total_seconds = total_time / 1_000_000
          bandwidth = (total_bytes / total_seconds) / 1_048_576  # MB/s


          {:ok, file} = File.open("latency_results_#{rank}.txt", [:append, :utf8])

          # bytes, ms, ms, mb/s
          write_to_file(file, "#{msg_size};#{Float.round(avg_rtt, 3)};#{Float.round(bandwidth, 2)}")

          IO.puts("Process #{rank}: Completed #{i}/#{sizes_count} #{byte_size(test_data)} iterations")


          # Close the file
          File.close(file)


          # results.append(
          # {msg_size, avg_rtt, bandwidth})
        end

        # # Write summary table to file
        # write_to_file(file, "\n==== Latency vs. Message Size Summary ====")
        # write_to_file(file, "Message Size (bytes) | Average RTT (ms) | Bandwidth (MB/s)")
        # write_to_file(file, "------------------------------------")

        # Enum.each(results, fn {size, rtt, bandwidth} ->
        #   write_to_file(file, "#{String.pad_leading(Integer.to_string(size), 18)} | #{Float.round(rtt, 3)} | #{Float.round(bandwidth, 2)}")
        # end)

        # write_to_file(file, "=======================================\n")

      1 ->
        # Process 1 (pong)
        sender_pid = Map.get(world, 0)

        # For each message size
        Enum.each(sizes, fn _expected_size ->
          # Get the current size we're testing
          {:ok, current_size, _from_pid, _tag} = ElixirMPI.MPI.recv(sender_pid, 0, 50000000)


          Enum.each(1..iterations, fn _i ->
            # Receive ping using the size as the tag
            {:ok, received_data, from_pid, tag} = ElixirMPI.MPI.recv(sender_pid, current_size, 50000000)

            # Send pong with the same tag
            ElixirMPI.MPI.send(received_data, 0, tag, world)
          end)
        end)
    end
  end

# Helper function to write to file
defp write_to_file(file, message) do
  IO.write(file, message <> "\n")
end

@doc """
  The function executed by each process in the scalability test.
  Processes are paired: 0-1, 2-3, 4-5, etc.
  """
  def scalability_process(rank, size, world, iterations, message_size) do
    # Determine if this process is a sender or receiver
    is_sender = rem(rank, 2) == 0

    # Calculate the partner process
    partner = if is_sender, do: rank + 1, else: rank - 1
    partner_pid = Map.get(world, partner)

    # Create test data
    test_data = generate_test_data(message_size)

    # Make sure all processes are ready before starting
    ElixirMPI.MPI.barrier(rank)

    if is_sender do
      # This is a sender process
      IO.puts("Process #{rank}: Starting sender, paired with #{partner}")

      start_time = :os.system_time(:milli_seconds)

      Enum.each(1..iterations, fn i ->
        # Send message
        ElixirMPI.MPI.send(test_data, partner, i, world)

        # Receive reply - use direct PID reference
        {:ok, _received_data, _from_pid, _tag} = ElixirMPI.MPI.recv(partner_pid, i, 50000000)

        # Print progress
        if rem(i, 100) == 0 do
          IO.puts("Process #{rank}: Completed #{i}/#{iterations} iterations with partner #{partner}")
        end
      end)

      end_time = :os.system_time(:milli_seconds)
      pair_time = end_time - start_time

      # Calculate and report pair-specific metrics
      IO.puts("Pair #{div(rank, 2)}: completed #{iterations} ping-pongs in #{pair_time} ms")
      IO.puts("Pair #{div(rank, 2)}: avg RTT: #{Float.round(pair_time / iterations, 3)} ms")
    else
      # This is a receiver process
      IO.puts("Process #{rank}: Starting receiver, paired with #{partner}")

      Enum.each(1..iterations, fn i ->
        # Receive message - use direct PID reference
        {:ok, received_data, _from_pid, _tag} = ElixirMPI.MPI.recv(partner_pid, i, 50000000)

        # Send reply
        ElixirMPI.MPI.send(received_data, partner, i, world)
      end)
    end

    # Synchronize all processes at the end
    ElixirMPI.MPI.barrier(rank)
  end
end
