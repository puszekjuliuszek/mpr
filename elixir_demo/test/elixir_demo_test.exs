defmodule ElixirMPITest do
  use ExUnit.Case
  doctest ElixirMPI

  test "greets the world" do
    assert ElixirMPI.hello() == :world
  end
end
