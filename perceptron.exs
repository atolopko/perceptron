defmodule Perceptron do

  def dot_prod(v1, v2) do
    Enum.zip(v1, v2)
    |> Enum.map(fn(e) -> elem(e, 0) * elem(e,1) end)
    |> Enum.sum
  end

  def vec_add(v1, v2) do
    Enum.zip(v1, v2)
    |> Enum.map(fn(e) -> elem(e, 0) + elem(e,1) end)
  end

  def vec_scale(v, n) do
    Enum.map(v, fn e -> e * n end)
  end

  def vec_to_s(v) do
    fmt = fn n -> Float.round((n * 1.0), 3) end
    Enum.map(v, fmt) |> Enum.join(",")
  end

  def sign(n) do
    if n < 0 do
      -1
    else
      1
    end
  end

  # Hypothosis function
  def h(x, w) do
    sign(dot_prod(w, [1 | x]))
  end

  def first_misclassified(d, w) do
    Enum.find(Map.keys(d), fn(x) -> h(x, w) != d[x] end)
  end

  # Perceptron learning algorithm
  # w(t+1) = w(t) + (y(t) * x(t))
  def improve_weights(w, x, y) do
    vec_add(w, vec_scale(x, y))
  end

  def normalize_weights(w) do
    div = List.last(w)
    if div == 0 do
      w
    else
      Enum.map(w, &(&1 / div))
    end
  end

  # d is the training data, mapping each input vector to -1/+1
  def learn_weights(d) do
    learn_weights(d, [0, 0, 0], 0)
  end

  def learn_weights(_, _, n) when n == 1024 do
    nil
  end
  
  def learn_weights(d, w, n) do
    case first_misclassified(d, w) do
      nil -> w
      x ->
        IO.puts "n: #{n}\tw: #{vec_to_s(w)}\tnorm_w: #{vec_to_s(normalize_weights(w))}\t" <>
          "dot_prod=#{dot_prod(w, [1 | x])}" <>
          "\tmisclassified: #{vec_to_s(x)} -> #{h(x, w)}"
        learn_weights(d, improve_weights(w, [1 | x], d[x]), n + 1)
    end
  end

  def verify(d, h) do
    Map.new(Enum.map(d,
          fn(e) -> { elem(e, 0), h.(elem(e, 0)) } end))
  end
end

defmodule Example do
  def training_data do
    %{ [1, 6] => -1,
       [1, 1] => -1,
       [4, 2] => -1,
       [3, 7] => 1,
       [5, 6] => 1,
       [8, 2] => 1 }
  end

  def run do
    w = Perceptron.learn_weights(training_data)
    if w do
      h = &Perceptron.h(&1, w)
      Perceptron.verify(training_data, h)
      Perceptron.normalize_weights(w)
    end
  end
end

