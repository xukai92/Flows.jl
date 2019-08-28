# Make some functions CUDA friendly

import CuArrays

CuArrays.@cufunc softplus(x) = x < 9 ? log1p(exp(x)) : x < 16 ? x + exp(-x) : x
CuArrays.@cufunc invsoftplus(x) = x <= 9 ? log(expm1(x)) : x <= 16 ? x - exp(-x) : x

CuArrays.@cufunc logit(x) = log(x) - log(1 - x)
CuArrays.@cufunc logistic(x) = inv(exp(-x) + 1)

# Fix for broadcast ^

CuArrays.culiteral_pow(::typeof(^), x::T, ::Val{0}) where {T<:Real} = one(x)
CuArrays.culiteral_pow(::typeof(^), x::T, ::Val{1}) where {T<:Real} = x
CuArrays.culiteral_pow(::typeof(^), x::T, ::Val{2}) where {T<:Real} = x * x
CuArrays.culiteral_pow(::typeof(^), x::T, ::Val{3}) where {T<:Real} = x * x * x
CuArrays.culiteral_pow(::typeof(^), x::T, ::Val{p}) where {T<:Real,p} = CUDAnative.pow(x, Int32(p))

# Distributions

rand(d::DiagNormal{T}, n::Int=1) where {T1,T2,TC<:CuArrays.CuArray,T<:Union{TC,Flux.TrackedArray{T1,T2,TC}}} = (randn(Float32, length(d.μ), n) |> Flux.gpu) .* sqrt.(softplus.(d.realσsq)) .+ d.μ
