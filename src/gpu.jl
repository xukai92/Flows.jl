using CuArrays: @cufunc
import CuArrays: culiteral_pow

@cufunc logit(x) = log(x) - log(1 - x)
@cufunc logistic(x) = inv(exp(-x) + 1)


import Flux: cpu, gpu

gpu(d::MvNormal01) = MvNormal01{:gpu}(d.dim)
cpu(d::MvNormal01) = MvNormal01{:cpu}(d.dim)

rand(d::MvNormal01{:gpu}, n::Int=1) = randn(Float32, d.dim, n) |> Flux.gpu


# Fix for broadcast ^
culiteral_pow(::typeof(^), x::T, ::Val{0}) where {T<:Real} = one(x)
culiteral_pow(::typeof(^), x::T, ::Val{1}) where {T<:Real} = x
culiteral_pow(::typeof(^), x::T, ::Val{2}) where {T<:Real} = x * x
culiteral_pow(::typeof(^), x::T, ::Val{3}) where {T<:Real} = x * x * x
culiteral_pow(::typeof(^), x::T, ::Val{p}) where {T<:Real,p} = CUDAnative.pow(x, Int32(p))