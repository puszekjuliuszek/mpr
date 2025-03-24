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
    {:ok, worker_pids}
  end

  @impl true
  def handle_cast({:start, n}, worker_pids) do

    #generate a and b
    a = Matrex.random(n)
    b = Matrex.random(n)

    for pid <- worker_pids, do: Worker.start(pid, a, b)

    {:noreply, worker_pids}
  end

  @impl true
  def handle_cast({:done, _c}, worker_pids) do
    BlasBenchmark.stop()
    {:noreply, worker_pids}
  end

  def start(n) do
    GenServer.cast(__MODULE__, {:start,n})
  end

  def finish(c) do
    GenServer.cast(__MODULE__, {:done, c})
  end
end
