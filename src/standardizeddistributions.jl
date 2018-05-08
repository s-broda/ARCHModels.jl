abstract type StandardizedDistribution{T<:AbstractFloat} <: Sampleable{Univariate, Continuous} end

mean(::StandardizedDistribution{T}) where {T} = zero(T)
var(::StandardizedDistribution{T}) where {T} = one(T)

struct StdNormal{T} <: StandardizedDistribution{T}
    StdNormal{T}() where {T} = new{T}()
end
StdNormal(T::Type=Float64) = StdNormal{T}()
rand(::StdNormal{T}) where {T} = randn(T)
logpdf(::StdNormal{T}, x::T) where {T} = normlogpdf(x)
nparams(::StdNormal) = 0

function constraints(::StdNormal{T}) where {T}
    lower = T[]
    upper = T[]
    return lower, upper
end

struct StdTDist{T} <: StandardizedDistribution{T}
    ν::T
    StdTDist{T}(ν::T) where {T} = (ν>2 ? new{T}(ν) : error("degrees of freedom must be greater than 2."))
end
StdTDist(ν::T) where {T} = StdTDist{T}(ν)
StdTDist(ν::Integer) = StdTDist(float(ν))
rand(d::StdTDist) = tdistrand(d.ν)*sqrt((d.ν-2)/d.ν)
logpdf(d::StdTDist{T}, x::T) where {T} = (s = sqrt((d.ν-2)/d.ν); x=x/s; - log(s) + lgamma((d.ν + 1) / 2) - log(d.ν * pi) / 2 - lgamma(d.ν / 2) + (-(d.ν + 1) / 2) * log(1 + x^2 / d.ν))
nparams(::StdTDist) = 1

function constraints(::StdTDist{T}) where {T}
    lower = T[2]
    upper = T[Inf]
    return lower, upper
end
