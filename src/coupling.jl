### Affine coupling layer from (Dinh et al., 2017)

abstract type AbstractAffineCoupling <: AbstractInvertibleTransformation end

# Affine coupling layer with a single function computing s and t

struct AffineCoupling{T,TM} <: AbstractAffineCoupling
    st::T
    mask::TM
end

function computest(t::AffineCoupling, input)
    dim = size(input, 1)
    st = t.st(input)
    return (s=logistic.(st[1:dim,:] .+ 2), t=st[dim+1:end,:])
end
computes(t::AffineCoupling, input) = computest(t, input).s

# Affine coupling layer with two functions computing s and t

struct AffineCouplingSlow{T1,T2,TM} <: AbstractAffineCoupling
    s::T1
    t::T2
    mask::TM
end

function computes(t::AffineCouplingSlow, input)
    temp = t.s(input) .+ eltype(input)(2)
    return logistic.(temp)
end

computest(t::AffineCouplingSlow, input) = (s=computes(t, input), t=t.t(input))

logabsdetjacob(
    t::T, 
    x; 
    s=computes(t, t.mask .* x)
) where {T<:AbstractAffineCoupling} = (invmask = 1 .- t.mask; sum(invmask .* s; dims=1))

function forward(t::T, x) where {T<:AbstractAffineCoupling}
    mask = t.mask
    invmask = 1 .- mask
    x_masked = mask .* x
    st = computest(t, x_masked)
    temp1 = exp.(st.s)
    temp2 = invmask .* (x .* temp1 + st.t)
    y = x_masked .+ temp2
    return (rv=y, logabsdetjacob=logabsdetjacob(t, nothing; s=st.s))
end

function forward(it::Inversed{T}, y) where {T<:AbstractAffineCoupling}
    t = inv(it)
    mask = t.mask
    invmask = 1 .- mask
    y_masked = mask .* y
    st = computest(t, y_masked)
    temp1 = (y - st.t)
    temp2 = exp.(-st.s)
    temp3 = temp1 .* temp2
    temp4 = invmask .* temp3
    x = y_masked .+ temp4
    return (rv=x, logabsdetjacob=-logabsdetjacob(t, nothing; s=st.s))
end

### Masking methods

abstract type AbstractMasking end

instantiate(::T) where {T<:AbstractMasking} = error("`create(::$T)` is not implemented.")

struct AlternatingMasking{T<:Function} <: AbstractMasking
    dim::Int
    is1::T
end
AlternatingMasking(dim) = AlternatingMasking(dim, i -> i % 2 == 0)

instantiate(masking::AlternatingMasking) = Int[masking.is1(i) for i in 1:masking.dim]