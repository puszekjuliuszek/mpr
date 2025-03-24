# import Head

defmodule BlasBenchmark do
  use Application
  use GenServer

  @impl true
  def start(_type, args) do
    {n, _} = Integer.parse(Enum.at(args, 0))
    start_link(4); # start with 4 workers
    start(n)
    {:ok, self()}
  end

  def start_link(worker_count) do
    GenServer.start_link(
      __MODULE__,
      {worker_count},
      name: __MODULE__
    )
  end

  # Callbacks
  @impl true
  def init({worker_count}) do
    worker_pids = for _ <- 1..worker_count do
      {_, pid} = Worker.start_link()
      pid
    end
    start_time = System.os_time(:millisecond)
    Head.start_link(worker_pids)
    Tail.start_link(length(worker_pids))

    {:ok, {worker_count, start_time}}
  end

  @impl true
  def handle_cast({:start, n}, {worker_count, _}) do
    start_time = System.os_time(:millisecond)
    Head.start(n)
    {:noreply, {worker_count, start_time}}
  end

  @impl true
  def handle_cast(:stop, {_, start_time}) do
    end_time = System.os_time(:millisecond)
    IO.puts("Calculations took: #{end_time - start_time} milliseconds")
    System.stop()
  end

  def start(n) do
    GenServer.cast(__MODULE__, {:start, n})
  end

  def stop() do
    GenServer.cast(__MODULE__, :stop)
  end
end
