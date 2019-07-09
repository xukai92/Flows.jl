struct Logit{T<:Real} <: AbstractInvertibleTransformation
    a::T
    b::T
end

logabsdetjacob(t::Logit{<:Real}, x) = log.((x .- t.a) .* (t.b .- x) ./ (t.b .- t.a))

forward(t::Logit, x) = (rv=logit.((x .- t.a) ./ (t.b .- t.a)), logabsdetjacob=-logabsdetjacob(t, x))

function forward(it::Inversed{Logit{T}}, y) where {T<:Real}
    t = it.original
    x = (t.b - t.a) * logistic.(y) .+ t.a
    return (rv=x, logabsdetjacob=logabsdetjacob(t, x))
end
