using CuArrays: @cufunc

@cufunc logit(x) = log(x) - log(1 - x)

import Flux: cpu, gpu

gpu(d::MvNormal01) = MvNormal01{:gpu}(d.dim)
cpu(d::MvNormal01) = MvNormal01{:cpu}(d.dim)

rand(d::MvNormal01{:gpu}, n::Int=1) = randn(Float32, d.dim, n) |> Flux.gpu