using Test, Flows
# using CuArrays

@testset "Tests" begin
    @testset "Abstraction" begin
        # Check composition by composing one transformation and its inverse
        t1 = Logit(0, 1)
        t2 = inv(t1)
        ct = compose(t1, t2)
        x = rand()
        @test forward(ct, x).rv ≈ x

        # Check composing composed transformation
        ct2 = compose(t1, t2, ct)
        @test forward(ct2, x).rv ≈ x
    end

    include("logit.jl")
    include("coupling.jl")
end