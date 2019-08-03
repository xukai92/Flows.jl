# Make some functions CUDA friendly

using CuArrays: @cufunc

@cufunc logit(x) = log(x) - log(1 - x)
@cufunc logistic(x) = inv(exp(-x) + 1)

# Fix for broadcast ^

import CuArrays

CuArrays.culiteral_pow(::typeof(^), x::T, ::Val{0}) where {T<:Real} = one(x)
CuArrays.culiteral_pow(::typeof(^), x::T, ::Val{1}) where {T<:Real} = x
CuArrays.culiteral_pow(::typeof(^), x::T, ::Val{2}) where {T<:Real} = x * x
CuArrays.culiteral_pow(::typeof(^), x::T, ::Val{3}) where {T<:Real} = x * x * x
CuArrays.culiteral_pow(::typeof(^), x::T, ::Val{p}) where {T<:Real,p} = CUDAnative.pow(x, Int32(p))

# Distributions

Flux.gpu(d::DiagNormal) = DiagNormal{:gpu}(Flux.gpu(d.μ), Flux.gpu(d.logσ))
Flux.cpu(d::DiagNormal) = DiagNormal{:cpu}(Flux.cpu(d.μ), Flux.cpu(d.logσ))

rand(d::DiagNormal{:gpu,TP}, n::Int=1) where {TP} = (randn(Float32, length(d.μ), n) |> Flux.gpu) .* exp.(d.logσ) .+ d.μ

Flux.gpu(d::MixtureModel) = MixtureModel(d.n_mixtures, Flux.gpu(d.logit_weights), Flux.gpu.(d.components))
Flux.cpu(d::MixtureModel) = MixtureModel(d.n_mixtures, Flux.cpu(d.logit_weights), Flux.cpu.(d.components))