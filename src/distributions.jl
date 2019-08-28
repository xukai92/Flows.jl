# Normal distribution with diagonal covariance that is easy to work with GPU

struct DiagNormal{T}
    μ::T
    realσsq::T
end

# The constant below is a hack to make things work on GPU.
const LOG2PI32 = log(2Float32(pi))

function DiagNormal(μ; σ=ones(eltype(μ), size(μ)))
    return DiagNormal(μ, invsoftplus.(σ .^ 2))
end

function logpdf(d::DiagNormal, x)
    diff = x .- d.μ
    diffsq = diff .* diff
    σsq = softplus.(d.realσsq)
    logσsq = log.(σsq)
    return sum(-(LOG2PI32 .+ logσsq .+ diffsq ./ σsq); dims=1) ./ 2
end

rand(d::DiagNormal, n::Int=1) = randn(Float32, length(d.μ), n) .* sqrt.(softplus.(d.realσsq)) .+ d.μ

# Mixture models

struct MixtureModel{TW,TC}
    # Number of mixtures
    n_mixtures::Int
    # Weights in log space
    logit_weights::TW
    # Actual components
    components::TC
end

function MixtureModel(components...; learn_weights=true)
    n_mixtures = length(components)
    logit_weights = zeros(Float32, n_mixtures)
    learn_weights && (logit_weights = logit_weights |> Flux.param)
    TW = typeof(logit_weights)
    TC = typeof(components)
    return MixtureModel{TW,TC}(n_mixtures, logit_weights, components)
end

compute_log_weights(d::MixtureModel) = d.logit_weights .- logsumexp(d.logit_weights; dims=:)

function logpdf(d::MixtureModel, x)
    log_probs = vcat(logpdf.(d.components, Ref(x))...)
    log_weights = compute_log_weights(d)
    log_weightedprobs = log_probs .+ log_weights
    return logsumexp(log_weightedprobs; dims=1)
end

function rand(d::MixtureModel, n::Int=1)
    log_weights = compute_log_weights(d)
    x = []
    for _ in 1:n
        i = rand(Categorical(exp.(log_weights)))
        push!(x, rand(d.components[i], 1))
    end
    return hcat(x...)
end

Flux.mapchildren(f, d::MixtureModel) = MixtureModel(d.n_mixtures, f(d.logit_weights), f.(d.components))
Flux.children(d::MixtureModel) = tuple(d.n_mixtures, d.logit_weights, d.components...)