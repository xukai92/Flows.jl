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
    return (s=st[1:dim,:], t=st[dim+1:end,:])
end
computes(t::AffineCoupling, input) = computest(t, input).s

# Affine coupling layer with two functions computing s and t

struct AffineCouplingSlow <: AbstractAffineCoupling
    s
    t
    mask
end

computest(t::AffineCouplingSlow, input) = (s=t.s(input), t=t.t(input))
computes(t::AffineCouplingSlow, input) = t.s(input)

logabsdetjacob(
    t::T, 
    x; 
    exps=exp.(computes(t, t.mask .* x))
) where {T<:AbstractAffineCoupling} = sum(exps; dims=1)

function forward(t::T, x) where {T<:AbstractAffineCoupling}
    mask = t.mask
    x_masked = mask .* x
    st = computest(t, x_masked)
    exps = exp.(st.s)
    y = x_masked + (1 .- mask) .* (x .* exps + st.t)
    return (rv=y, logabsdetjacob=logabsdetjacob(t, nothing; exps=exps))
end

function forward(it::Inversed{T}, y) where {T<:AbstractAffineCoupling}
    t = it.original; mask = t.mask
    y_masked = mask .* y
    st = computest(t, y_masked)
    invexps = exp.(-st.s)
    x = y_masked + (1 .- mask) .* (y - st.t) .* invexps
    return (rv=x, logabsdetjacob=-logabsdetjacob(t, nothing; exps=1 ./ invexps))
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