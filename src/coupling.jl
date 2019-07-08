struct AffineCoupling <: AbstractInvertibleTransformation
    s
    t
    mask
end

logabsdetjacob(
    t::AffineCoupling, 
    x; 
    exps=exp.(t.s(t.mask .* x))
) = sum(exps; dims=1)

function forward(t::AffineCoupling, x)
    mask = t.mask
    x_masked = mask .* x
    exps = exp.(t.s(x_masked))
    y = x_masked + (1 .- mask) .* (x .* exps + t.t(x_masked))
    return (rv=y, logabsdetjacob=logabsdetjacob(t, nothing; exps=exps))
end

function forward(it::Inversed{AffineCoupling}, y)
    t = it.original; mask = t.mask
    y_masked = mask .* y
    invexps = exp.(-t.s(y_masked))
    x = y_masked + (1 .- mask) .* (y - t.t(y_masked)) .* invexps
    return (rv=x, logabsdetjacob=-logabsdetjacob(t, nothing; exps=1 ./ invexps))
end