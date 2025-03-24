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
  def handle_cast(:start, _) do
    IO.puts("[MYLOG] worker starts... #{inspect(self())}")
    c = Matrex.random(100)
    #what 20 operations???

    IO.puts("[MYLOG] worker starts finishes... #{inspect(self())}")
    Tail.send_result()
    {:noreply, {}}
  end

  def start(pid) do
    GenServer.cast(pid, :start)
  end
end
