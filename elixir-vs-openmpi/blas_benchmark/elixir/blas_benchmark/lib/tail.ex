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
    {:ok, {worker_count, finished}}
  end

  @impl true
  def handle_cast({:send_result, c}, {worker_count, finished}) do
    if finished + 1 == worker_count do
      Head.finish(c)
    end
    {:noreply, {worker_count, finished + 1}}
  end

  def send_result(c) do
    GenServer.cast(__MODULE__, {:send_result, c})
  end
end
