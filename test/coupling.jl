using Test, Flows
using Distributions: Bernoulli
using Flux: Flux, Tracker, Chain, Dense, relu, gpu, jacobian
using LinearAlgebra: det

function test_logabsdetjacob(t, x; atol=1f-6)
    jacob_ad = jacobian(_x -> forward(t, _x).rv, x)
    jacob_impl = forward(t, x).logabsdetjacob[1]
    @test jacob_impl ≈ log(abs(det(jacob_ad))) atol=atol
end

function test_AffineCoupling(t, x; test_jacob=true)
    # Test sizes
    res = forward(t, x)
    y = res.rv
    @test size(res.rv) == size(x)
    @test size(res.logabsdetjacob) == (1, size(x, 2))
    
    # Test if inverse works
    it = inv(t)
    res2 = forward(it, y)
    @test Tracker.data(res2.rv) ≈ x atol=1f-3
    @test Tracker.data(res.logabsdetjacob) ≈ -Tracker.data(res2.logabsdetjacob) atol=1f-3
    
    # Test logabsdetjacob computation
    if test_jacob
        x1 = x[:,1]
        test_logabsdetjacob(t, x1)
        test_logabsdetjacob(it, x1)
    end
end

@testset "AffineCouplingSlow" begin
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
    test_AffineCoupling(t_slow, x)
    
    f2 = Dense(h, 2d) |> gpu
    t = AffineCoupling(Chain(f1, f2), mask)
    test_AffineCoupling(t, x; test_jacob=false)
end