defmodule Head do
  use GenServer

  def start_link(worker_pids) do
    GenServer.start_link(
      __MODULE__,
      worker_pids,
      name: __MODULE__
    )
  end

  # Callbacks
  @impl true
  def init(worker_pids) do
    IO.puts("Initializing head with workers")
    {:ok, worker_pids}
  end

  @impl true
  def handle_cast(:start, worker_pids) do
    IO.puts("[MYLOG] starting operation calculating operations");

    for pid <- worker_pids, do: Worker.start(pid)
    {:noreply, worker_pids}
  end

  @impl true
  def handle_cast(:done, worker_pids) do
    IO.puts("[MYLOG] finished all operations");
    BlasBenchmark.stop()
    {:noreply, worker_pids}
  end

  def start() do
    GenServer.cast(__MODULE__, :start)
  end

  def finish() do
    GenServer.cast(__MODULE__, :done)
  end
end
