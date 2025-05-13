defmodule ElixirMPI.MPI do
  @moduledoc """
  An implementation of common MPI (Message Passing Interface) functions in Elixir.
  """

  @doc """
  Initializes the MPI environment with a given function to execute on each process.

  ## Parameters
  - size: number of processes to create
  - func: function to execute on each process, should accept rank as argument

  ## Returns
  - {rank, size, world} for the calling process
  """
  def init(size, func \\ nil) when is_integer(size) and size > 0 do
    # Create a coordinator process
    coordinator_pid = spawn(fn -> coordinator_loop(%{}, size) end)
    Process.register(coordinator_pid, :mpi_coordinator)

    # Spawn worker processes
    processes = Enum.map(0..(size-1), fn rank ->
      spawn(fn ->
        Process.flag(:trap_exit, true)
        # Register with coordinator
        send(coordinator_pid, {:register, rank, self()})

        # Wait for world map
        world = receive do
          {:world_map, world_map} -> world_map
        end

        IO.puts("Process #{rank}: Received world map")

        # Wait a bit to ensure all processes are ready
        :timer.sleep(100)

        # Execute the provided function if given
        if func do
          IO.puts("Process #{rank}: Starting function execution")
          func.(rank, size, world)
          IO.puts("Process #{rank}: Function execution completed")
        end

        # Send exit signal to coordinator
        send(coordinator_pid, {:EXIT, self(), :normal})
      end)
    end)

    # Create world map
    world = Enum.with_index(processes)
      |> Enum.map(fn {pid, rank} -> {rank, pid} end)
      |> Map.new()

    # Send world map to all processes
    Enum.each(processes, fn pid ->
      send(pid, {:world_map, world})
    end)

    # Let the main process know about the world
    {0, size, world}
  end

  @doc """
  Finalizes the MPI environment and waits for all processes to complete.

  ## Returns
  - :ok when all processes have completed
  """
  def finalize(timeout \\ 1000000) do
    coordinator_pid = Process.whereis(:mpi_coordinator)

    if coordinator_pid == nil do
      IO.puts("Finalize error: MPI coordinator not found!")
      {:error, :no_coordinator}
    else
      IO.puts("Finalizing MPI: Waiting for all processes to complete")
      # Send finalize signal to coordinator
      send(coordinator_pid, {:finalize, self()})

      # Wait for coordinator to confirm all processes have completed
      receive do
        {:finalize_complete} ->
          IO.puts("MPI finalization completed")

          # Unregister and terminate the coordinator
          Process.unregister(:mpi_coordinator)
          Process.exit(coordinator_pid, :normal)
          :ok
      after
        timeout ->
          IO.puts("MPI finalization timeout")
          {:error, :finalize_timeout}
      end
    end
  end

  @doc """
  Sends a message from the current process to a specified destination process.
  """
  def send(data, dest, tag, world) when is_map(world) and is_integer(dest) do
    case Map.get(world, dest) do
      nil ->
        IO.puts("Send error: Invalid destination #{dest}")
        {:error, :invalid_destination}
      dest_pid ->
        IO.puts("Sending message to process #{dest}")
        Kernel.send(dest_pid, {:mpi_message, self(), tag, data})
        :ok
    end
  end

  @doc """
  Receives a message sent by another process.
  """
  def recv(source \\ :any, tag \\ :any, timeout \\ :infinity) do
    IO.puts("Waiting to receive message from #{inspect(source)} with tag #{inspect(tag)}")
    receive do
      {:mpi_message, from_pid, msg_tag, data} ->
        IO.puts("Received message from #{inspect(from_pid)} with tag #{msg_tag}")
        source_match = source == :any or
                      (is_pid(source) and source == from_pid) or
                      (is_integer(source) and get_rank_from_pid(from_pid) == source)
        tag_match = tag == :any or tag == msg_tag

        if source_match and tag_match do
          IO.puts("Message matches criteria, returning")
          {:ok, data, from_pid, msg_tag}
        else
          IO.puts("Message doesn't match criteria, putting back in mailbox")
          send(self(), {:mpi_message, from_pid, msg_tag, data})
          recv(source, tag, timeout)
        end
    after
      timeout ->
        IO.puts("Receive timeout after #{timeout}ms")
        {:error, :timeout}
    end
  end

  @doc """
  Implements a barrier synchronization.
  """
  def barrier(rank) do
    IO.puts("Process #{rank}: Entering barrier")
    coordinator = Process.whereis(:mpi_coordinator)

    if coordinator == nil do
      IO.puts("Error: MPI coordinator not found!")
      {:error, :no_coordinator}
    else
      # Notify coordinator we've reached the barrier
      IO.puts("Process #{rank}: Notifying coordinator of barrier")
      send(coordinator, {:barrier_reached, rank})

      # Wait for barrier completion signal
      IO.puts("Process #{rank}: Waiting for barrier completion")
      receive do
        {:barrier_complete} ->
          IO.puts("Process #{rank}: Barrier completed")
          :ok
      after
        5000 ->
          IO.puts("Process #{rank}: Barrier timeout")
          {:error, :barrier_timeout}
      end
    end
  end

  @doc """
  Broadcasts a message from the root process to all other processes.
  """
  def bcast(data, root, rank, world) do
    IO.puts("Process #{rank}: In broadcast, root=#{root}")
    if rank == root do
      # Root sends data to all other processes
      IO.puts("Process #{rank}: I am root, sending to all others")
      Enum.each(Map.keys(world), fn dest_rank ->
        if dest_rank != root do
          IO.puts("Process #{rank}: Broadcasting to #{dest_rank}")
          send(data, dest_rank, 0, world)
        end
      end)
      data
    else
      # Non-root processes receive data
      IO.puts("Process #{rank}: I am not root, waiting for broadcast")
      case recv(root, 0, 5000) do
        {:ok, received_data, _, _} ->
          IO.puts("Process #{rank}: Received broadcast data")
          received_data
        {:error, reason} ->
          IO.puts("Process #{rank}: Error receiving broadcast: #{reason}")
          nil
      end
    end
  end

  # Helper functions

  defp get_rank_from_pid(pid) do
    coordinator = Process.whereis(:mpi_coordinator)
    if coordinator == nil do
      IO.puts("Error: Coordinator not found when looking up rank!")
      nil
    else
      send(coordinator, {:get_rank, pid, self()})

      receive do
        {:rank_result, rank} -> rank
      after
        1000 ->
          IO.puts("Timeout when looking up rank for PID #{inspect(pid)}")
          nil
      end
    end
  end

  defp coordinator_loop(state, size) do
    new_state =
      receive do
        {:register, rank, pid} ->
          updated_state = Map.put(state, rank, pid)
          IO.puts("Coordinator: Registered process #{rank}, #{map_size(updated_state)}/#{size} processes")
          updated_state

        {:get_rank, pid, reply_to} ->
          # Find rank by PID
          rank_entry = Enum.find(state, {nil, nil}, fn {_, p} -> p == pid end)
          rank = elem(rank_entry, 0)
          IO.puts("Coordinator: Looking up rank for PID #{inspect(pid)}: #{inspect(rank)}")
          send(reply_to, {:rank_result, rank})
          state

        {:barrier_reached, rank} ->
          # Track processes that have reached the barrier
          barrier_procs = Map.get(state, :barrier_procs, MapSet.new())
          updated_barrier = MapSet.put(barrier_procs, rank)
          IO.puts("Coordinator: Process #{rank} reached barrier. Count: #{MapSet.size(updated_barrier)}/#{size}")

          # If all processes have reached the barrier, release them
          if MapSet.size(updated_barrier) >= size do
            IO.puts("Coordinator: All processes reached barrier. Releasing barrier.")
            Enum.each(state, fn {_rank, process_pid} ->
              if is_pid(process_pid), do: send(process_pid, {:barrier_complete})
            end)
            Map.delete(state, :barrier_procs)
          else
            Map.put(state, :barrier_procs, updated_barrier)
          end

        {:EXIT, pid, reason} ->
          IO.puts("Coordinator: Process #{inspect(pid)} exited with reason: #{inspect(reason)}")
          # Find and remove the exited process from the state
          {exited_rank, remaining_processes} = Enum.reduce(state, {nil, %{}}, fn
            {rank, ^pid}, {_, acc} -> {rank, acc}
            {rank, process_pid}, {exited, acc} when is_pid(process_pid) -> {exited, Map.put(acc, rank, process_pid)}
            {key, value}, {exited, acc} -> {exited, Map.put(acc, key, value)}
          end)

          if exited_rank != nil do
            IO.puts("Coordinator: Removed process with rank #{exited_rank}")
          end

          # Check if all processes have exited
          remaining_process_count = Enum.count(remaining_processes, fn {key, val} ->
            is_integer(key) and is_pid(val)
          end)

          if remaining_process_count == 0 do
            # All MPI processes have exited
            IO.puts("Coordinator: All processes have exited")
            # Check if there's a finalize call waiting
            case Map.get(remaining_processes, :finalize_caller) do
              nil -> remaining_processes
              caller_pid when is_pid(caller_pid) ->
                IO.puts("Coordinator: Notifying finalize caller that all processes completed")
                send(caller_pid, {:finalize_complete})
                Map.delete(remaining_processes, :finalize_caller)
            end
          else
            remaining_processes
          end

        {:finalize, caller_pid} ->
          IO.puts("Coordinator: Received finalize request")
          # Store the caller PID to notify when all processes exit
          updated_state = Map.put(state, :finalize_caller, caller_pid)

          # Check if all processes have already exited
          process_count = Enum.count(updated_state, fn {key, val} ->
            is_integer(key) and is_pid(val)
          end)

          if process_count == 0 do
            IO.puts("Coordinator: All processes already completed, notifying finalize caller")
            send(caller_pid, {:finalize_complete})
            Map.delete(updated_state, :finalize_caller)
          else
            IO.puts("Coordinator: Waiting for #{process_count} processes to complete")
            updated_state
          end

        other ->
          IO.puts("Coordinator: Received unknown message: #{inspect(other)}")
          state
      end

    coordinator_loop(new_state, size)
  end
end
