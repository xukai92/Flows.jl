function (a::Flux.Dense)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    res = W * x
    res = res .+ b
    return σ.(res)
end