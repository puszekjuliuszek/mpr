defmodule MontecarloElixirTest do
  use ExUnit.Case
  doctest MontecarloElixir

  test "greets the world" do
    assert MontecarloElixir.hello() == :world
  end
end
