
defmodule ElixirMPI do
  use Application

  def start(_type, _args) do
    # node_number = 4
    # run(node_number)
    ElixirMPI.Benchmark.ping_pong(1000, 1024)
    # ElixirMPI.Benchmark.latency_vs_size([8, 64, 512, 4096, 32768, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 1073741824, 2147483647], 1000)
    # ElixirMPI.Benchmark.scalability_test()
    Supervisor.start_link([], [strategy: :one_for_one])
    end

  def run(n) do
    IO.puts("Starting MPI run")
    # Initialize MPI with processes that execute the example function
    {rank, size, world} = ElixirMPI.MPI.init(n, &main/3)

    IO.puts("Main process: Initialized MPI with rank=#{rank}, size=#{size}")
    IO.puts("Main process: World map: #{inspect(world)}")

    # Wait for all processes to complete and finalize MPI
    case ElixirMPI.MPI.finalize() do
      :ok ->
        IO.puts("Main process: MPI finalized successfully")
        # Exit the program after finalization
      {:error, reason} ->
        IO.puts("Main process: MPI finalization failed: #{reason}")
    end
  end

  # The function to execute on each MPI process
  def main(rank, size, world) do
    IO.puts("Process #{rank}: Starting example function with size=#{size}")

    # Basic communication example
    if rank == 0 do
      # Process 0 sends a message to process 1
      data = "Hello from process 0"
      ElixirMPI.MPI.send(data, 1, 42, world)
      IO.puts("Process #{rank}: Sent '#{data}' to process 1")
    end

    if rank == 1 do
      # Process 1 receives the message
      case ElixirMPI.MPI.recv(:any, :any, 5000) do
        {:ok, data, source, tag} ->
          IO.puts("Process #{rank}: Received '#{data}' from #{inspect(source)} with tag #{tag}")
        {:error, reason} ->
          IO.puts("Process #{rank}: Error receiving message: #{reason}")
      end
    end

    # Demonstrate barrier
    IO.puts("Process #{rank}: Reaching barrier")
    ElixirMPI.MPI.barrier(rank)
    IO.puts("Process #{rank}: Passed barrier")

    # Demonstrate broadcast
    broadcast_data = if rank == 0, do: "Broadcast message", else: nil
    result = ElixirMPI.MPI.bcast(broadcast_data, 0, rank, world)
    IO.puts("Process #{rank}: Received broadcast: #{result}")
  end
end
