################################################################################
#general functions
"""
    mean(spec::MeanSpec)
Return the mean implied by MeanSpec
"""
mean(spec::MeanSpec) = mean(spec, spec.coefs)
################################################################################
#NoIntercept
"""
    NoIntercept{T} <: MeanSpec{T}

A mean specification without an intercept (i.e., the mean is zero).
"""
struct NoIntercept{T} <: MeanSpec{T}
    coefs::Vector{T}
end
"""
    NoIntercept(T::Type=Float64)
    NoIntercept{T}()
    NoIntercept(v::Vector)

Create an instance of NoIntercept.
"""
NoIntercept{T}(coef::Vector{T}, data) where {T} = NoIntercept(coef)
NoIntercept(T::Type=Float64) = NoIntercept(T[])
NoIntercept{T}() where {T} = NoIntercept(T[])
nparams(::NoIntercept) = 0
coefnames(::NoIntercept) = String[]

function constraints(::Type{<:NoIntercept}, ::Type{T})  where {T<:AbstractFloat}
    lower = T[]
    upper = T[]
    return lower, upper
end

function startingvals(::NoIntercept{T}, data)  where {T<:AbstractFloat}
    return T[]
end

@inline function mean(::NoIntercept, meancoefs::Vector{T}) where {T}
    return zero(T)
end

################################################################################
#Intercept
"""
    Intercept{T} <: MeanSpec{T}

A mean specification with just an intercept.
"""
struct Intercept{T} <: MeanSpec{T}
    coefs::Vector{T}
end

"""
    Intercept(mu)

Create an instance of Intercept. `mu` can be passed as a scalar or vector.
"""
Intercept{T}(mu::Vector{T}, data) where {T} = Intercept(mu)
Intercept(mu) = Intercept([mu])
Intercept(mu::Integer) = Intercept(float(mu))
nparams(::Intercept) = 1
coefnames(::Intercept) = ["μ"]

function constraints(::Type{<:Intercept}, ::Type{T})  where {T<:AbstractFloat}
    lower = T[-Inf]
    upper = T[Inf]
    return lower, upper
end

function startingvals(::Intercept, data::Vector{T})  where {T<:AbstractFloat}
    return T[mean(data)]
end

@inline function mean(::Intercept, meancoefs::Vector{T}) where {T}
    return @inbounds meancoefs[1]
end
