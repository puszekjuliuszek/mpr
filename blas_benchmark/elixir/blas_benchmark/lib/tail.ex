defmodule Tail do
  use GenServer

  def start_link(worker_count) do
    GenServer.start_link(
      __MODULE__,
      {worker_count, 0},
      name: __MODULE__
    )
  end

  # Callbacks
  @impl true
  def init({worker_count, finished}) do
    IO.puts("Initializing tail with workers pids")
    {:ok, {worker_count, finished}}
  end

  @impl true
  def handle_cast(:send_result, {worker_count, finished}) do
    if finished + 1 == worker_count do
      Head.finish()
    end
    {:noreply, {worker_count, finished + 1}}
  end

  def send_result() do
    GenServer.cast(__MODULE__, :send_result)
  end
end
