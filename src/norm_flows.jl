using Distributions
using LinearAlgebra
using Random
using Flux

abstract type AbstractFlow <: AbstractInvertibleTransformation end
# abstract type AbstractFlow end
abstract type AbstractPlanarLayer <: AbstractFlow end

mutable struct PlanarLayer <: AbstractPlanarLayer
    w
    u
    u_hat
    b
end

function update_u_hat(u, w)
    # to preserve invertibility
    u_hat = u + (m(transpose(w)*u) - transpose(w)*u)[1]*w/(norm(w[:,1],2)^2)
end

function update_u_hat!(flow::PlanarLayer)
    flow.u_hat = flow.u + (m(transpose(flow.w)*flow.u) - transpose(flow.w)*flow.u)[1]*flow.w/(norm(flow.w,2)^2)
end


function PlanarLayer(dims::Int)
    w = param(randn(dims, 1))
    u = param(randn(dims, 1))
    b = param(randn(1))
    u_hat = update_u_hat(u, w)
    return PlanarLayer(w, u, u_hat, b)
end

f(z, flow::PlanarLayer) = z + flow.u_hat*tanh.(transpose(flow.w)*z .+ flow.b)
m(x) = -1 .+ log.(1 .+ exp.(x))
dtanh(x) = 1 .- (tanh.(x)).^2
ψ(z, w, b) = dtanh(transpose(w)*z .+ b).*w

function forward(flow::T, z) where {T<:AbstractPlanarLayer}
    update_u_hat!(flow)
    # compute log_det_jacobian
    transformed = f(z, flow)
    psi = ψ(transformed, flow.w, flow.b)
    log_det_jacobian = log.(abs.(1.0 .+ transpose(psi)*flow.u_hat))

    return (rv=transformed, logabsdetjacob=log_det_jacobian)
end
