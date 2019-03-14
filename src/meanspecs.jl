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

@inline presample(::NoIntercept) = 0
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

@inline presample(::Intercept) = 0

################################################################################
#ARMA
struct ARMA{p, q, T} <: MeanSpec{T}
    coef::Vector{T}
end
nparams(::ARMA) = p+q+1
function coefnames(::ARMA{p, q}) where {p, q}
    names = Array{String, 1}(undef, p+q+1)
    names[1] = "μ"
    names[2:p+1] .= (i -> "φ"*subscript(i)).([1:p...])
    names[2+p:p+q+1] .= (i -> "θ"*subscript(i)).([1:q...])

    return names
end
@inline presample(::ARMA{p, q}) where {p, q} = max(p, q)
