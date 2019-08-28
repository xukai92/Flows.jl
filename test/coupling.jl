using Test, Flows
using Distributions: Bernoulli
using Flux: Chain, Dense, relu, gpu

include("common.jl")

@testset "AbstractAffineCoupling" begin
    d = 5
    n = 2
    x = randn(Float32, d, n) |> gpu
    
    p = 0.5
    mask = rand(Bernoulli(p), d) |> gpu
    
    h = 20
    f1 = Dense(d, h, relu) |> gpu
    f21 = Dense(h, d) |> gpu
    f22 = Dense(h, d) |> gpu
    t_slow = AffineCouplingSlow(Chain(f1, f21), Chain(f1, f22), mask)
    test_invtrans(t_slow, x)
    
    f2 = Dense(h, 2d) |> gpu
    t = AffineCoupling(Chain(f1, f2), mask)
    test_invtrans(t, x; test_jacob=false)
end