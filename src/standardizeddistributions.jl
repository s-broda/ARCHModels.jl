################################################################################
#general functions

#loop invariant part of the kernel
@inline kernelinvariants(::Type{<:StandardizedDistribution}, coefs) = ()

"""
    Standardized{D<:ContinuousUnivariateDistribution, T}  <: StandardizedDistribution{T}
A wrapper type for standardizing a distribution from Distributions.jl.
"""
struct Standardized{D<:ContinuousUnivariateDistribution, T}  <: StandardizedDistribution{T}
    coefs::Vector{T}
end
Standardized{D, T}(coefs::AbstractFloat...) where {D, T} = Standardized{D, T}([coefs...])
(::Type{Standardized{D, T1} where T1})(coefs::Array{T}) where {D, T} = Standardized{D, T}(coefs)
(::Type{Standardized{D, T1} where T1})(coefs::T...) where {D, T} = Standardized{D, T}([coefs...])
rand(s::Standardized{D, T}) where {D, T} = (rand(D(s.coefs...))-mean(D(s.coefs...)))./std(D(s.coefs...))
@inline logkernel(S::Type{<:Standardized{D, T1} where T1}, x, coefs::Vector{T}) where {D, T} = (try sig=std(D(coefs...)); logpdf(D(coefs...), mean(D(coefs...)) + sig*x)+log(sig); catch; T(-Inf); end)
@inline logconst(S::Type{<:Standardized{D, T1} where T1}, coefs::Vector{T}) where {D, T} = zero(T)
nparams(S::Type{<:Standardized{D, T} where T}) where {D} = length(fieldnames(D))
coefnames(S::Type{<:Standardized{D, T}}) where {D, T} = [string.(fieldnames(D))...]
distname(S::Type{<:Standardized{D, T}}) where {D, T} = D{T}.name
function quantile(s::Standardized{D, T}, q::Real) where {D, T}
    (quantile(D(s.coefs...), q)-mean(D(s.coefs...)))./std(D(s.coefs...))
end


function constraints(S::Type{<:Standardized{D, T1} where T1}, ::Type{T})  where {D, T}
    lower = Vector{T}(undef, nparams(S))
    upper = Vector{T}(undef, nparams(S))
    fill!(lower, T(-Inf))
    fill!(upper, T(Inf))
    lower, upper
end

function startingvals(S::Type{<:Standardized{D, T1} where {T1}}, data::Vector{T})  where {T, D}
    svals = Vector{T}(undef, nparams(S))
    fill!(svals, eps(T))
    return svals
end

#for rand to work
Base.eltype(::StandardizedDistribution{T}) where {T} = T

"""
    fit(::Type{SD}, data; algorithm=BFGS(), kwargs...)

Fit a standardized distribution to the data, using the MLE. Keyword arguments
are passed on to the optimizer.
"""
function fit(::Type{SD}, data::Vector{T};
             algorithm=BFGS(), kwargs...
             ) where {SD<:StandardizedDistribution, T<:AbstractFloat}
    nparams(SD) == 0 && return SD{T}()
    obj = x -> -loglik(SD, data, x)
    lower, upper = constraints(SD, T)
    x0 = startingvals(SD, data)
    res = optimize(obj, lower, upper, x0, Fminbox(algorithm); kwargs...)
    coefs = res.minimizer
    return SD(coefs)
end


function loglik(::Type{SD}, data::Vector{<:AbstractFloat},
                coefs::Vector{T2}
                ) where {SD<:StandardizedDistribution, T2}
    T = length(data)
    length(coefs) == nparams(SD) || throw(NumParamError(nparams(SD), length(coefs)))
    @inbounds begin
        LL = zero(T2)
        @fastmath for t = 1:T
            LL += logkernel(SD, data[t], coefs)
        end#for
    end#inbounds
    LL += T*logconst(SD, coefs)
end#function

################################################################################
#StdNormal
"""
    StdNormal{T} <: StandardizedDistribution{T}

The standard Normal distribution.
"""
struct StdNormal{T} <: StandardizedDistribution{T}
    coefs::Vector{T}
end
"""
    StdNormal(T::Type=Float64)
    StdNormal(v::Vector)
    StdNormal{T}()

Construct an instance of StdNormal.
"""
StdNormal(T::Type=Float64) = StdNormal(T[])
StdNormal{T}() where {T} = StdNormal(T[])
rand(::StdNormal{T}) where {T} = randn(T)
@inline logkernel(::Type{<:StdNormal}, x, coefs) = -abs2(x)/2
@inline logconst(::Type{<:StdNormal}, coefs::Vector{T}) where {T} =  -T(log2π)/2
nparams(::Type{<:StdNormal}) = 0
coefnames(::Type{<:StdNormal}) = String[]
distname(::Type{<:StdNormal}) = "Gaussian"

function constraints(::Type{<:StdNormal}, ::Type{T})  where {T<:AbstractFloat}
    lower = T[]
    upper = T[]
    return lower, upper
end

function startingvals(::Type{<:StdNormal}, data::Vector{T})  where {T<:AbstractFloat}
    return T[]
end

function quantile(::StdNormal, q::Real)
    norminvcdf(q)
end

################################################################################
#StdTDist

"""
    StdTDist{T} <: StandardizedDistribution{T}

The standardized (mean zero, variance one) Student's t distribution.
"""
struct StdTDist{T} <: StandardizedDistribution{T}
    coefs::Vector{T}
end

"""
    StdTDist(v)

Create a standardized t distribution with `v` degrees of freedom. `ν`` can be passed
as a scalar or vector.
"""
StdTDist(ν) = StdTDist([ν])
StdTDist(ν::Integer) = StdTDist(float(ν))
(rand(d::StdTDist{T})::T) where {T}  =  (ν=d.coefs[1]; tdistrand(ν)*sqrt((ν-2)/ν))
@inline logkernel(::Type{<:StdTDist}, x, coefs) = (-(coefs[1] + 1) / 2) * log1p(abs2(x) / (coefs[1]-2))
@inline logconst(::Type{<:StdTDist}, coefs)  = (lgamma((coefs[1] + 1) / 2)
                                               - log((coefs[1]-2) * pi) / 2
                                               - lgamma(coefs[1] / 2)
                                               )
nparams(::Type{<:StdTDist}) = 1
coefnames(::Type{<:StdTDist}) = ["ν"]
distname(::Type{<:StdTDist}) = "Student's t"

function constraints(::Type{<:StdTDist}, ::Type{T}) where {T}
    lower = T[20/10]
    upper = T[Inf]
    return lower, upper
end

function startingvals(::Type{<:StdTDist}, data::Array{T}) where {T}
    #mean of abs(t)
    eabst(ν)=2*sqrt(ν-2)/(ν-1)/beta(ν/2, 1/2)
    ##alteratively, could use mean of log(abs(t)):
    #elogabst(ν)=log(ν-2)/2-digamma(ν/2)/2+digamma(1/2)/2
    ht = T[]
    lht = T[]
    zt = T[]
    loglik!(ht, lht, zt, GARCH{1, 1}, StdNormal, Intercept, data, vcat(startingvals(GARCH{1, 1}, data), startingvals(Intercept, data)))
    lower = convert(T, 2)
    upper = convert(T, 30)
    z = mean(abs.(data.-mean(data))./sqrt.(ht))
    z > eabst(upper) ? [upper] : [find_zero(x -> z-eabst(x), (lower, upper))]
end

function quantile(dist::StdTDist, q::Real)
    tdistinvcdf(dist.coefs..., q)
end

################################################################################
#StdGED
#Nardon, M. and Pianca, P. (2009). Simulation techniques for generalized Gaussian densities. Journal
#of Statistical Software and Simulation, 79(11):1317–1329.

"""
    StdGED{T} <: StandardizedDistribution{T}

The standardized (mean zero, variance one) generalized error distribution.
"""
struct StdGED{T} <: StandardizedDistribution{T}
    coefs::Vector{T}
end

"""
    StdGED(p)

Create a standardized generalized error distribution parameter `p`. `p` can be passed
as a scalar or vector.
"""
StdGED(p) = StdGED([p])


(rand(d::StdGED{T})::T) where {T} = (p = d.coefs[1]; ip=1/p;  (2*rand()-1)*gammarand(1+ip, 1)^ip * sqrt(gamma(ip) / gamma(3*ip)) )


@inline logconst(::Type{<:StdGED}, coefs)  = (p = coefs[1]; ip = 1/p; lgamma(3*ip)/2 - lgamma(ip)*3/2 - logtwo  - log(ip))
@inline logkernel(::Type{<:StdGED}, x, coefs, s) = (p = coefs[1]; -abs(x*s)^p)
@inline kernelinvariants(::Type{<:StdGED}, coefs) = (p = coefs[1]; ip = 1/p; (sqrt(gamma(3*ip) / gamma(ip)),))

nparams(::Type{<:StdGED}) = 1
coefnames(::Type{<:StdGED}) = ["p"]
distname(::Type{<:StdGED}) = "Generalized Error Distribution"

function constraints(::Type{<:StdGED}, ::Type{T}) where {T}
    lower = [zero(T)]
    upper = T[Inf]
    return lower, upper
end

function startingvals(::Type{<:StdGED}, data::Array{T}) where {T}
    T[1]
end

function quantile(dist::StdGED, q::Real)
end
