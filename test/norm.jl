using Test, Flows
using Flux: gpu, testmode!

include("common.jl")

@testset "InvertibleBatchNorm" begin
    d = 5
    n = 2
    x = randn(Float32, d, n) |> gpu

    t = InvertibleBatchNorm(d) |> gpu
    testmode!(t, true)
    test_invtrans(t, x; test_jacob=false)
end