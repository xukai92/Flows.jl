### Affine coupling layer from (Dinh et al., 2017)

abstract type AbstractAffineCoupling <: AbstractInvertibleTransformation end

# Affine coupling layer with a single function computing s and t

struct AffineCoupling <: AbstractAffineCoupling
    st
    mask
end

function computest(t::AffineCoupling, input)
    dim = size(input, 1)
    st = t.st(input)
    return (s=logistic.(st[1:dim,:] .+ 2) .+ 1f-3, t=st[dim+1:end,:])
end
computes(t::AffineCoupling, input) = computest(t, input).s

# Affine coupling layer with two functions computing s and t

struct AffineCouplingSlow <: AbstractAffineCoupling
    s
    t
    mask
end

computes(t::AffineCouplingSlow, input) = logistic.(t.s(input) .+ 2) .+ 1f-3
computest(t::AffineCouplingSlow, input) = (s=computes(t, input), t=t.t(input))

logabsdetjacob(
    t::T, 
    x; 
    s=computes(t, t.mask .* x)
) where {T<:AbstractAffineCoupling} = sum((1 .- t.mask) .* s; dims=1)

function forward(t::T, x) where {T<:AbstractAffineCoupling}
    mask = t.mask
    x_masked = mask .* x
    st = computest(t, x_masked)
    y = x_masked + (1 .- mask) .* (x .* exp.(st.s) + st.t)
    return (rv=y, logabsdetjacob=logabsdetjacob(t, nothing; s=st.s))
end

function forward(it::Inversed{T}, y) where {T<:AbstractAffineCoupling}
    t = inv(it); mask = t.mask
    y_masked = mask .* y
    st = computest(t, y_masked)
    x = y_masked + (1 .- mask) .* (y - st.t) .* exp.(-st.s)
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

instantiate(masking::AlternatingMasking)::Vector{Bool} = [masking.is1(i) for i in 1:masking.dim]