# import Head

defmodule BlasBenchmark do
  use Application
  use GenServer

  @impl true
  def start(_type, _args) do
    IO.puts("test does it work...")
    res = start_link(5, 10);

    start()
    {:ok, self()}
    # res
  end

  def start_link(worker_count, matrix_size) do
    GenServer.start_link(
      __MODULE__,
      {worker_count, matrix_size, 0},
      name: __MODULE__
    )
  end

  # Callbacks
  @impl true
  def init({worker_count, matrix_size, start_time}) do
    IO.puts("Initializing head, tail and workers...")
    worker_pids = for _ <- 1..worker_count do
      {_, pid} = Worker.start_link()
      pid
    end

    IO.puts("[MYLOG] workers #{inspect(worker_pids)}")

    start_time = System.os_time(:millisecond)
    Head.start_link(worker_pids)
    Tail.start_link(length(worker_pids))

    {:ok, {worker_count, matrix_size, start_time}}
  end

  @impl true
  def handle_cast(:start, {worker_count, matrix_size, start_time}) do
    start_time = System.os_time(:millisecond)
    Head.start()
    {:noreply, {worker_count, matrix_size, start_time}}
  end

  @impl true
  def handle_cast(:stop, {worker_count, matrix_size, start_time}) do
    end_time = System.os_time(:millisecond)
    IO.puts("Calculations took: #{end_time - start_time} milliseconds")
    System.stop()
  end

  def start() do
    GenServer.cast(__MODULE__, :start)
  end

  def stop() do
    IO.puts("finishing blas in bechmark")
    GenServer.cast(__MODULE__, :stop)
  end
end
