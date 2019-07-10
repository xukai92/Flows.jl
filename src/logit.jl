# TODO: make this compatible with neural layers and GPUs

struct Logit{T<:Real} <: AbstractInvertibleTransformation
    a::T
    b::T
end

logabsdetjacob(t::Logit{<:Real}, x) = sum(log.((x .- t.a) .* (t.b .- x) ./ (t.b .- t.a)); dims=1)

forward(t::Logit, x) = (rv=logit.((x .- t.a) ./ (t.b .- t.a)), logabsdetjacob=-logabsdetjacob(t, x))

function forward(it::Inversed{Logit{T}}, y) where {T<:Real}
    t = inv(it)
    x = (t.b - t.a) * logistic.(y) .+ t.a
    return (rv=x, logabsdetjacob=logabsdetjacob(t, x))
end
