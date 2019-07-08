using Flows
using Distributions: rand, Bernoulli
using Flux: Tracker, Chain, Dense, relu, gpu

function test_AffineCoupling()
    d = 5
    n = 2
    p = 0.5
    mask = rand(Bernoulli(p), d) |> gpu
    h = 20
    f1 = Dense(d, h, relu) |> gpu
    f21 = Dense(h, d) |> gpu
    f22 = Dense(h, d) |> gpu
    t = AffineCoupling(Chain(f1, f21), Chain(f1, f22), mask)
    x = randn(d,n) |> gpu
    res = forward(t, x)
    y = res.rv
    @test size(res.rv) == size(x)
    @test size(res.logabsdetjacob) == (1,n)
    it = inv(t)
    res2 = forward(it, y)
    @test Tracker.data(res2.rv) ≈ x
    @test Tracker.data(res.logabsdetjacob) ≈ -Tracker.data(res2.logabsdetjacob)
end

@testset "AffineCoupling" begin
    test_AffineCoupling()
end