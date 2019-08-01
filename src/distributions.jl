# Define a MvNormal distribution that is easy to work with GPU
struct MvNormal01{T}
    dim::Int
end
MvNormal01(dim::Int) = MvNormal01{:cpu}(dim)

# The constant below is a hack to make things work on GPU.
const LOG2PI32 = log(2Float32(pi))
function logpdf(d::MvNormal01, x)
    temp1 = x .* x
    temp2 = LOG2PI32 .+ temp1
    temp3 = -temp2
    temp4 = sum(temp3; dims=1)
    return temp4 ./ eltype(temp4)(2)
end
rand(d::MvNormal01{:cpu}, n::Int=1) = randn(Float32, d.dim, n)