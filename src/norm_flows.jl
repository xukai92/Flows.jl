using Distributions
using MLToolkit
using LinearAlgebra
using Random

abstract type AbstractFlow end

abstract type AbstractPlanarFlow <:AbstractFlow end

struct PlanarFlow <: AbstractPlanarFlow
    wₖ
    uₖ
    u_hat
    bₖ
    mu
    sig
    depth
end

function update_u_hat(uₖ, wₖ)
    # to preserve invertibility
    u_hat  = [[1.0,1.0] for i in 1:length(uₖ)]
    for i in 1:length(uₖ)
        u_hat[i] = uₖ[i] + (m(transpose(wₖ[i])*uₖ[i]) - transpose(wₖ[i])*uₖ[i])*wₖ[i]/(norm(wₖ[i],2)^2)
    end
    u_hat
end
function update_u_hat!(flow::PlanarFlow)
    for i in 1:flow.depth
        flow.u_hat[i] = flow.uₖ[i] + (m(transpose(flow.wₖ[i])*flow.uₖ[i]) - transpose(flow.wₖ[i])*flow.uₖ[i])*flow.wₖ[i]/(norm(flow.wₖ[i],2)^2)
    end
end

function PlanarFlow(dims::Int, depth::Int)
    wₖ = [randn(dims) for i in 1:depth]
    uₖ = [randn(dims) for i in 1:depth]
    bₖ = randn(depth)
    u_hat = update_u_hat(uₖ, wₖ)
    mu = [0.0 for i in 1:dims]
    sig = [1.0]
    return PlanarFlow(wₖ, uₖ, u_hat, bₖ, mu, sig, depth)
end
# PlanarFlow(wₖ, uₖ, bₖ) = PlanarFlow(wₖ, uₖ, update_u_hat(uₖ, wₖ), bₖ, length(wₖ))

# function getzₖ(fs, j, z)
#     zₖ[i]
# end
function planar_f(i, flow::PlanarFlow)
    u, w, b = flow.u_hat[i], flow.wₖ[i],flow.bₖ[i]
    f(z) = z + u*tanh.(transpose(w)*z + b)
end

m(x) = -1 + log(1+exp(x))
dtanh(x) = 1 - tanh.(x)^2
ψ(z, w, b) = dtanh(transpose(w)*z + b)*w
function forward(flow::T, z) where {T<:AbstractPlanarFlow}

    update_u_hat!(flow)

    # compute log_det_jacobian
    log_det_jacobian = 0
    prev = z
    for i in 1:flow.depth
        u, w, b = flow.u_hat[i], flow.wₖ[i],flow.bₖ[i]
        prev = planar_f(i, flow)(prev)
        psi = ψ(prev, w, b)
        log_det_jacobian += log.(abs.(1 .+ transpose(psi)*u))
    end

    return (rv=prev, logabsdetjacob=log_det_jacobian)
end

function logabsdetjacob(flow::T, z) where {T<:AbstractPlanarFlow}
    update_u_hat!(flow)

    # compute log_det_jacobian
    log_det_jacobian = 0
    prev = z
    for i in 1:flow.depth
        u, w, b = flow.u_hat[i], flow.wₖ[i],flow.bₖ[i]
        prev = planar_f(i, flow)(prev)
        psi = ψ(prev, w, b)
        log_det_jacobian += log.(abs.(1 .+ transpose(psi)*u))
    end

    return log_det_jacobian
end

function forward(flow::Inversed{T}, z) where {T<:AbstractPlanarFlow}
    # TODO: Implement
    0
end

PlanarFlow(2,6)
