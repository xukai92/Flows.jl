module Flows

using Requires
@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("gpu.jl")

import Base: inv

abstract type AbstractInvertibleTransformation end

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

logabsdetjacob(it::T1, y::T2) where {T<:AbstractInvertibleTransformation,T1<:Inversed{T},T2} = 
    error("`logabsdetjacob(it::$T1, y::$T2)` is not implemented.")
forward(it::T1, y::T2) where {T<:AbstractInvertibleTransformation,T1<:Inversed{T},T2} = 
    error("`forward(it::$T1, y::$T2)` is not implemented.")

# Composition

struct Composed{T<:AbstractInvertibleTransformation} <: AbstractInvertibleTransformation
    ts::Vector{T}
end

compose(ts...) = Composed([ts...])

inv(ct::Composed{T}) where {T<:AbstractInvertibleTransformation} = Composed(map(inv, ct.ts))

function forward(ct::Composed{<:AbstractInvertibleTransformation}, x)
    res = (rv=x, logabsdetjacob=0)
    for t in ct.ts
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

end # module
