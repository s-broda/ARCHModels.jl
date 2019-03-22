################################################################################
#general functions
# work around https://github.com/JuliaStats/Distributions.jl/issues/846
rand(::AbstractRNG, sd::StandardizedDistribution) where {T} = rand(sd)

#loop invariant part of the kernel
@inline kernelinvariants(::Type{<:StandardizedDistribution}, coefs) = ()
################################################################################
#standardized
"""
    Standardized{D<:ContinuousUnivariateDistribution, T}  <: StandardizedDistribution{T}
A wrapper type for standardizing a distribution from Distributions.jl.
"""
struct Standardized{D<:ContinuousUnivariateDistribution, T<:AbstractFloat}  <: StandardizedDistribution{T}
    coefs::Vector{T}
end
Standardized{D}(coefs::T...) where {D, T} = Standardized{D, T}([coefs...])
Standardized{D}(coefs::Vector{T}) where {D, T} = Standardized{D, T}([coefs...])
rand(s::Standardized{D, T}) where {D, T} = (rand(D(s.coefs...))-mean(D(s.coefs...)))./std(D(s.coefs...))
@inline logkernel(S::Type{<:Standardized{D, T1} where T1}, x, coefs::Vector{T}) where {D, T} = (try sig=std(D(coefs...)); logpdf(D(coefs...), mean(D(coefs...)) + sig*x)+log(sig); catch; T(-Inf); end)
@inline logconst(S::Type{<:Standardized{D, T1} where T1}, coefs::Vector{T}) where {D, T} = zero(T)
nparams(S::Type{<:Standardized{D, T} where T}) where {D} = length(fieldnames(D))
coefnames(S::Type{<:Standardized{D, T}}) where {D, T} = [string.(fieldnames(D))...]
distname(S::Type{<:Standardized{D, T}}) where {D, T} = string(D{T}.name)
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
        iv = kernelinvariants(SD, coefs)
        @fastmath for t = 1:T
            LL += logkernel(SD, data[t], coefs, iv...)
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
    function StdNormal{T}(coefs::Vector) where {T}
        length(coefs) == 0 || throw(NumParamError(0, length(coefs)))
        new{T}(coefs)
    end
end
"""
    StdNormal(T::Type=Float64)
    StdNormal(v::Vector)
    StdNormal{T}()

Construct an instance of StdNormal.
"""
StdNormal(T::Type{<:AbstractFloat}=Float64) = StdNormal(T[])
StdNormal{T}() where {T<:AbstractFloat} = StdNormal(T[])
StdNormal(v::Vector{T}) where {T} = StdNormal{T}(v)
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
#StdT

"""
    StdT{T} <: StandardizedDistribution{T}

The standardized (mean zero, variance one) Student's t distribution.
"""
struct StdT{T} <: StandardizedDistribution{T}
    coefs::Vector{T}
    function StdT{T}(coefs::Vector) where {T}
        length(coefs) == 1 || throw(NumParamError(1, length(coefs)))
        new{T}(coefs)
    end
end

"""
    StdT(v)

Create a standardized t distribution with `v` degrees of freedom. `ν`` can be passed
as a scalar or vector.
"""
StdT(ν) = StdT([ν])
StdT(ν::Integer) = StdT(float(ν))
StdT(v::Vector{T}) where {T} = StdT{T}(v)
(rand(d::StdT{T})::T) where {T}  =  (ν=d.coefs[1]; tdistrand(ν)*sqrt((ν-2)/ν))
@inline kernelinvariants(::Type{<:StdT}, coefs) = (1/ (coefs[1]-2),)
@inline logkernel(::Type{<:StdT}, x, coefs, iv) = (-(coefs[1] + 1) / 2) * log1p(abs2(x) *iv)
@inline logconst(::Type{<:StdT}, coefs)  = (lgamma((coefs[1] + 1) / 2)
                                               - log((coefs[1]-2) * pi) / 2
                                               - lgamma(coefs[1] / 2)
                                               )
nparams(::Type{<:StdT}) = 1
coefnames(::Type{<:StdT}) = ["ν"]
distname(::Type{<:StdT}) = "Student's t"

function constraints(::Type{<:StdT}, ::Type{T}) where {T}
    lower = T[20/10]
    upper = T[Inf]
    return lower, upper
end

function startingvals(::Type{<:StdT}, data::Array{T}) where {T}
    #mean of abs(t)
    eabst(ν)=2*sqrt(ν-2)/(ν-1)/beta(ν/2, 1/2)
    ##alteratively, could use mean of log(abs(t)):
    #elogabst(ν)=log(ν-2)/2-digamma(ν/2)/2+digamma(1/2)/2
    ht = T[]
    lht = T[]
    zt = T[]
    at = T[]
    loglik!(ht, lht, zt, at,  GARCH{1, 1}, StdNormal, Intercept(0.), data, vcat(startingvals(GARCH{1, 1}, data), startingvals(Intercept(0.), data)))
    lower = convert(T, 2)
    upper = convert(T, 30)
    z = mean(abs.(data.-mean(data))./sqrt.(ht))
    z > eabst(upper) ? [upper] : [find_zero(x -> z-eabst(x), (lower, upper))]
end

function quantile(dist::StdT, q::Real)
    v = dist.coefs[1]
    tdistinvcdf(v, q)*sqrt((v-2)/v)
end

################################################################################
#StdGED

"""
    StdGED{T} <: StandardizedDistribution{T}

The standardized (mean zero, variance one) generalized error distribution.
"""
struct StdGED{T} <: StandardizedDistribution{T}
    coefs::Vector{T}
    function StdGED{T}(coefs::Vector) where {T}
        length(coefs) == 1 || throw(NumParamError(1, length(coefs)))
        new{T}(coefs)
    end
end

"""
    StdGED(p)

Create a standardized generalized error distribution parameter `p`. `p` can be passed
as a scalar or vector.
"""
StdGED(p) = StdGED([p])
StdGED(p::Integer) = StdGED(float(p))
StdGED(v::Vector{T}) where {T} = StdGED{T}(v)

(rand(d::StdGED{T})::T) where {T} = (p = d.coefs[1]; ip=1/p;  (2*rand()-1)*gammarand(1+ip, 1)^ip * sqrt(gamma(ip) / gamma(3*ip)) )


@inline logconst(::Type{<:StdGED}, coefs)  = (p = coefs[1]; ip = 1/p; lgamma(3*ip)/2 - lgamma(ip)*3/2 - logtwo  - log(ip))
@inline logkernel(::Type{<:StdGED}, x, coefs, s) = (p = coefs[1]; -abs(x*s)^p)
@inline kernelinvariants(::Type{<:StdGED}, coefs) = (p = coefs[1]; ip = 1/p; (sqrt(gamma(3*ip) / gamma(ip)),))

nparams(::Type{<:StdGED}) = 1
coefnames(::Type{<:StdGED}) = ["p"]
distname(::Type{<:StdGED}) = "GED"

function constraints(::Type{<:StdGED}, ::Type{T}) where {T}
    lower = [zero(T)]
    upper = T[Inf]
    return lower, upper
end

function startingvals(::Type{<:StdGED}, data::Array{T}) where {T}
    ht = T[]
    lht = T[]
    zt = T[]
    at = T[]
    loglik!(ht, lht, zt, at, GARCH{1, 1}, StdNormal, Intercept(0.), data, vcat(startingvals(GARCH{1, 1}, data), startingvals(Intercept(0.), data)))
    z = mean((abs.(data.-mean(data))./sqrt.(ht)).^4)
    lower = T(0.05)
    upper = T(25.)
    f(r) = z-gamma(5/r)*gamma(1/r)/gamma(3/r)^2
    f(lower)>0 && return [lower]
    f(upper)<0 && return [upper]
    return T[find_zero(f, (lower, upper))]

end

function quantile(dist::StdGED, q::Real)
    p = dist.coefs[1]
    ip = 1/p
    qq = 2*q-1
    return sign(qq) * (gammainvcdf(ip, 1., abs(qq)))^ip/kernelinvariants(StdGED, [p])[1]
end
