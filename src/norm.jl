# Code adapted from Flux.jl:
# https://github.com/FluxML/Flux.jl/blob/68ba6e4e2fa4b86e2fef8dc6d0a5d795428a6fac/src/layers/normalise.jl#L117-L206
# License: https://github.com/FluxML/Flux.jl/blob/master/LICENSE.md

mutable struct InvertibleBatchNorm <: AbstractInvertibleTransformation
    μ  # moving mean
    σ² # moving std
    ϵ
    momentum
    active::Bool
end

InvertibleBatchNorm(chs::Int; ϵ=1f-5, momentum=0.1f0) = InvertibleBatchNorm(zeros(Float32, chs), ones(Float32, chs), ϵ, momentum, true)

function affinesize(x)
    dims = length(size(x))
    channels = size(x, dims-1)
    affinesize = ones(Int, dims)
    affinesize[end-1] = channels
    return affinesize
end

logabsdetjacob(
    t::T, 
    x; 
    σ²=reshape(t.σ², affinesize(x)...)
) where {T<:InvertibleBatchNorm} = -sum(log.(σ² .+ t.ϵ)) / 2 .* typeof(Flux.data(x))(ones(Float32, size(x, 2))')

function forward(t::T, x) where {T<:InvertibleBatchNorm} 
    @assert size(x, ndims(x) - 1) == length(t.μ) "`InvertibleBatchNorm` expected $(length(t.μ)) channels, got $(size(x, ndims(x) - 1))"
    as = affinesize(x)
    m = prod(size(x)[1:end-2]) * size(x)[end]
    if !t.active
        μ = reshape(t.μ, as...)
        σ² = reshape(t.σ², as...)
        ϵ = t.ϵ
    else
        Tx = eltype(x)
        dims = length(size(x))
        axes = [1:dims-2; dims] # axes to reduce along (all but channels axis)
        μ = mean(x, dims=axes)
        σ² = sum((x .- μ) .^ 2, dims=axes) ./ m
        ϵ = Flux.data(convert(Tx, t.ϵ))
        # Update moving mean/std
        mtm = Flux.data(convert(Tx, t.momentum))
        t.μ = (1 - mtm) .* t.μ .+ mtm .* reshape(Flux.data(μ), :)
        t.σ² = (1 - mtm) .* t.σ² .+ (mtm * m / (m - 1)) .* reshape(Flux.data(σ²), :)
    end
    y = (x .- μ) ./ sqrt.(σ² .+ ϵ)
    return (rv=y, logabsdetjacob=logabsdetjacob(t, x; σ²=σ²))
end

# TODO: make this function take kw argument `σ²`
logabsdetjacob(it::Inversed{T}, y) where {T<:InvertibleBatchNorm} = (x = y; -logabsdetjacob(it.original, x))
    
function forward(it::Inversed{T}, y) where {T<:InvertibleBatchNorm}
    t = inv(it)
    @assert t.active == false "`forward(::Inversed{InvertibleBatchNorm})` is only available in test mode but not in training mode."
    as = affinesize(y)
    μ = reshape(t.μ, as...)
    σ² = reshape(t.σ², as...)
    x = y .* sqrt.(σ² .+ t.ϵ) .+ μ
    return (rv=x, logabsdetjacob=logabsdetjacob(it, x))
end

# Flux support

Flux.mapchildren(f, t::InvertibleBatchNorm) = InvertibleBatchNorm(f(t.μ), f(t.σ²), t.ϵ, t.momentum, t.active)
Flux._testmode!(t::InvertibleBatchNorm, test) = (t.active = !test)