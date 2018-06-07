struct NoIntercept{T} <: MeanSpec{T}
    coefs::Vector{T}
    NoIntercept{T}() where {T} = new{T}(T[])
end
NoIntercept(T::Type=Float64) = NoIntercept{T}()

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
struct Intercept{T} <: MeanSpec{T}
    coefs::Vector{T}
    Intercept{T}(coefs) where {T} = new{T}([coefs])
end
Intercept(mu::T) where {T} = Intercept{T}(mu)
nparams(::Type{<:Intercept}) = 1
coefnames(::Type{<:Intercept}) = ["μ"]

function constraints(::Type{<:Intercept}, ::Type{T})  where {T<:AbstractFloat}
    lower = T[-Inf]
    upper = T[Inf]
    return lower, upper
end

function startingvals(::Type{<:Intercept}, data::Vector{T})  where {T<:AbstractFloat}
    return [mean(data)]
end

function mean(::Type{<:Intercept}, meancoefs::Vector{T}) where {T}
    return meancoefs[1]
end