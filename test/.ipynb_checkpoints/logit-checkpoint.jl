using Flows
using Distributions: logpdf, Truncated, Normal
using Bijectors: logpdf_with_trans
using StatsFuns: logit
using Flux: gpu
using CuArrays

function test_Logit()
    # x -> y and y -> x
    a, b = 1, 3
    d = Truncated(Normal(0, 1), a, b)
    t = Logit(a, b)
    y = randn()
    it = inv(t)
    res = forward(it, y)
    x = res.rv
    # Test log-abs-Jacob against Bijectors.jl
    @test logpdf(d, x) + res.logabsdetjacob == logpdf_with_trans(d, x, true)
    res2 = forward(t, x)
    @test y ≈ res2.rv
    
    # Test on both CPU and GPU
    t = Logit(0, 1)
    x = [0.3, 0.4]
    y_true = logit.(x)
    @test forward(t, x).rv ≈ y_true
    x = x |> gpu
    @test forward(t, x).rv ≈ y_true
end

@testset "Logit" begin
    test_Logit()
end