using Flux, CuArrays, BenchmarkTools

conv = Conv((2,2), 1=>2, relu)
x = rand(28,28,1,200)

@info "CPU"
@benchmark conv(x)

conv = conv |> gpu
x = x |> gpu

@info "GPU"
@benchmark conv(x)
