# Define a MvNormal distribution that is easy to work with GPU
struct MvNormal01{T}
    dim::Int
end
MvNormal01(dim::Int) = MvNormal01{:cpu}(dim)

# The constant below is a hack to make things work on GPU.
const LOG2PI32 = log(2Float32(pi))
logpdf(d::MvNormal01, x) = sum(-(LOG2PI32 .+ x .* x) ./ 2; dims=1)
rand(d::MvNormal01{:cpu}, n::Int=1) = randn(Float32, d.dim, n)