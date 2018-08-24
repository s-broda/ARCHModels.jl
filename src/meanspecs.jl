"""
    NoIntercept{T} <: MeanSpec{T}

A mean specification without an intercept (i.e., the mean is zero).
"""
struct NoIntercept{T} <: MeanSpec{T}
    coefs::Vector{T}
end
"""
    NoIntercept(a)
    NoIntercept{T}()

Create an instance of NoIntercept. `a` is either a type or an empty array.
"""
NoIntercept(T::Type=Float64) = NoIntercept(T[])
NoIntercept{T}() where {T} = NoIntercept(T[])
nparams(::Type{<:NoIntercept}) = 0
coefnames(::Type{<:NoIntercept}) = String[]

function constraints(::Type{<:NoIntercept}, ::Type{T})  where {T<:AbstractFloat}
    lower = T[]
    upper = T[]
    return lower, upper
end

function startingvals(::Type{<:NoIntercept}, data::Vector{T})  where {T<:AbstractFloat}
    return T[]
end

function mean(::Type{<:NoIntercept}, meancoefs::Vector{T}) where {T}
    return zero(T)
end

"""
    Intercept{T} <: MeanSpec{T}

A mean specification with just an intercept.
"""
struct Intercept{T} <: MeanSpec{T}
    coefs::Vector{T}
end

"""
    Intercept(mu)

Create an instance of Intercept. `mu` can be a 1-element vector or a scalar.    
"""
Intercept(mu) = Intercept([mu])
Intercept(mu::Integer) = Intercept(float(mu))
nparams(::Type{<:Intercept}) = 1
coefnames(::Type{<:Intercept}) = ["μ"]

function constraints(::Type{<:Intercept}, ::Type{T})  where {T<:AbstractFloat}
    lower = T[-Inf]
    upper = T[Inf]
    return lower, upper
end

function startingvals(::Type{<:Intercept}, data::Vector{T})  where {T<:AbstractFloat}
    return T[mean(data)]
end

function mean(::Type{<:Intercept}, meancoefs::Vector{T}) where {T}
    return meancoefs[1]
end
