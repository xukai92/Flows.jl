module Flows


using Requires
@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("gpu.jl")

import Base: inv
import Flux
import Distributions: rand, logpdf

### Abstractions

abstract type AbstractInvertibleTransformation end

# NOTE: The second argument `x` is assumed to be the **input** of the transformation `t` (t: x -> y).
logabsdetjacob(t::T1, x::T2) where {T1<:AbstractInvertibleTransformation,T2} = 
    error("`logabsdetjacob(t::$T1, x::$T2)` is not implemented.")
forward(t::T1, x::T2) where {T1<:AbstractInvertibleTransformation,T2} = 
    error("`forward(t::$T1, x::$T2)` is not implemented.")

# Inverse

struct Inversed{T<:AbstractInvertibleTransformation} <: AbstractInvertibleTransformation
    original::T
end

inv(t::T) where {T<:AbstractInvertibleTransformation} = Inversed(t)
inv(it::Inversed{T}) where {T<:AbstractInvertibleTransformation} = it.original

# NOTE: The second argument `y` is assumed to be the **input** of the transformation `it` (it: y -> x).
logabsdetjacob(it::T1, y::T2) where {T<:AbstractInvertibleTransformation,T1<:Inversed{T},T2} = 
    error("`logabsdetjacob(it::$T1, y::$T2)` is not implemented.")
forward(it::T1, y::T2) where {T<:AbstractInvertibleTransformation,T1<:Inversed{T},T2} = 
    error("`forward(it::$T1, y::$T2)` is not implemented.")

# Composition

struct Composed{T<:AbstractInvertibleTransformation} <: AbstractInvertibleTransformation
    ts::Vector{T}
end

compose(ts...) = Composed([ts...])

inv(ct::Composed{T}) where {T<:AbstractInvertibleTransformation} = Composed(map(inv, reverse(ct.ts)))

function forward(ct::Composed{<:AbstractInvertibleTransformation}, x)
    # Evaluate the first transform to init `res` so that 
    # we avoid possible type instability issues, which would happen
    # especially using GPUs.
    res = forward(ct.ts[1], x)
    for t in ct.ts[2:end]
        res′ = forward(t, res.rv)
        res = (rv=res′.rv, logabsdetjacob=res.logabsdetjacob + res′.logabsdetjacob)
    end
    return res
end

export AbstractInvertibleTransformation, logabsdetjacob, forward, 
       Inversed, inv, 
       Composed, compose

# Logit transformation

using StatsFuns: logistic, logit
include("logit.jl")
export Logit

# Affine coupling transformation

include("coupling.jl")
export AffineCoupling, AffineCouplingSlow, AbstractMasking, AlternatingMasking, instantiate

# Batch normalisation transformation

using Statistics: mean
include("norm.jl")
export InvertibleBatchNorm

# Distributions

include("distributions.jl")
export MvNormal01

# Make all transformations callable.
# This has to be done in this manner because
# we cannot add method to abstract types.

for T in [Inversed, Composed, Logit, AffineCoupling, InvertibleBatchNorm]
    @eval (t::$T)(x) = forward(t, x)
end

### Flow
# Note: the flow abstraction is a draft and subject to change

struct Flow{T<:AbstractInvertibleTransformation}
    t::T
    base
end

# TODO: figure out what's the best way to do below
rand(f::Flow{T}, n::Int=1) where {T<:AbstractInvertibleTransformation} = f.t(rand(f.base, n::Int)).rv

# We cannot use broadcast as that doesn't work with 
# multivariate random variables.
function logpdf(f::Flow{T}, x) where {T<:AbstractInvertibleTransformation}
    it = inv(f.t)
    res = forward(it, x)
    return logpdf(f.base, res.rv) + res.logabsdetjacob
end

export Flow, rand, logpdf

### Flux support

Flux.children(t::T) where {T<:AbstractInvertibleTransformation} = map(pn -> getfield(t, pn), propertynames(t))
Flux.mapchildren(f, t::T) where {T<:AbstractInvertibleTransformation} = T(f.(Flux.children(ct))...)

Flux.@treelike(Inversed)

Flux.children(ct::Composed{T}) where {T<:AbstractInvertibleTransformation} = tuple(ct.ts...)
Flux.mapchildren(f, ct::Composed{T}) where {T<:AbstractInvertibleTransformation} = compose(f.(Flux.children(ct))...)

Flux.@treelike(AffineCoupling)
Flux.@treelike(AffineCouplingSlow)

Flux.@treelike(Flow)


end # module
