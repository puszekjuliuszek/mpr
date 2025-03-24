defmodule Worker do
  use GenServer

  def start_link() do
    GenServer.start_link(
      __MODULE__,
      {}
    )
  end

  @impl true
  def init(_) do
    {:ok, {}}
  end

  @impl true
  def handle_cast({:start, a, b}, _) do
    c = Matrex.multiply(a, b)

    Tail.send_result(c)
    {:noreply, {}}
  end

  def start(pid, a, b) do
    GenServer.cast(pid, {:start, a, b})
  end
end
