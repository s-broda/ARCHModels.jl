"""
    MultivariateStdNormal{T, d} <: MultivariateStandardizedDistribution{T, d}
The multivariate standard normal distribution.
"""
struct MultivariateStdNormal{T, d} <: MultivariateStandardizedDistribution{T, d}
    coefs::Vector{T}
end
MultivariateStdNormal{T, d}() where {T, d} = MultivariateStdNormal{T, d}(T[])
MultivariateStdNormal(T::Type, d::Int) = MultivariateStdNormal{T, d}()
MultivariateStdNormal(v::Vector{T}, d::Int) where {T} = MultivariateStdNormal{T, d}()
MultivariateStdNormal{T}(d::Int) where {T} = MultivariateStdNormal{T, d}()
MultivariateStdNormal(d::Int) = MultivariateStdNormal{Float64, d}(Float64[])
rand(::MultivariateStdNormal{T, d}) where {T, d} = randn(T, d)
nparams(::Type{<:MultivariateStdNormal}) = 0
coefnames(::Type{<:MultivariateStdNormal}) = String[]
distname(::Type{<:MultivariateStdNormal}) = "Multivariate Normal"
