################################################################################
#NoIntercept
"""
    NoIntercept{T} <: MeanSpec{T}

A mean specification without an intercept (i.e., the mean is zero).
"""
struct NoIntercept{T} <: MeanSpec{T}
    coefs::Vector{T}
    function NoIntercept{T}(coefs::Vector) where {T}
        length(coefs) == 0 || throw(NumParamError(0, length(coefs)))
        new{T}(coefs)
    end
end
"""
    NoIntercept(T::Type=Float64)
    NoIntercept{T}()
    NoIntercept(v::Vector)

Create an instance of NoIntercept.
"""
NoIntercept(coefs::Vector{T}) where {T} = NoIntercept{T}(coefs)
NoIntercept(T::Type=Float64) = NoIntercept(T[])
NoIntercept{T}() where {T} = NoIntercept(T[])
nparams(::Type{<:NoIntercept}) = 0
coefnames(::NoIntercept) = String[]

function constraints(::Type{<:NoIntercept}, ::Type{T})  where {T<:AbstractFloat}
    lower = T[]
    upper = T[]
    return lower, upper
end

function startingvals(::NoIntercept{T}, data)  where {T<:AbstractFloat}
    return T[]
end

Base.@propagate_inbounds @inline function mean(
    at, ht, lht, data, meanspec::NoIntercept{T}, meancoefs, t
    ) where {T}
    return zero(T)
end


@inline presample(::NoIntercept) = 0

Base.@propagate_inbounds @inline function uncond(::NoIntercept{T}) where {T}
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
    function Intercept{T}(coefs::Vector) where {T}
        length(coefs) == 1 || throw(NumParamError(1, length(coefs)))
        new{T}(coefs)
    end
end

"""
    Intercept(mu)

Create an instance of Intercept. `mu` can be passed as a scalar or vector.
"""
Intercept(coefs::Vector{T}) where {T} = Intercept{T}(coefs)
Intercept(mu) = Intercept([mu])
Intercept(mu::Integer) = Intercept(float(mu))
nparams(::Type{<:Intercept}) = 1
coefnames(::Intercept) = ["μ"]

function constraints(::Type{<:Intercept}, ::Type{T})  where {T<:AbstractFloat}
    lower = T[-Inf]
    upper = T[Inf]
    return lower, upper
end

function startingvals(::Intercept, data::Vector{T})  where {T<:AbstractFloat}
    return T[mean(data)]
end

Base.@propagate_inbounds @inline function mean(
    at, ht, lht, data, meanspec::Intercept{T}, meancoefs, t
    ) where {T}
    return meancoefs[1]
end

@inline presample(::Intercept) = 0

Base.@propagate_inbounds @inline function uncond(m::Intercept)
    return m.coefs[1]
end
################################################################################
#ARMA
"""
    ARMA{p, q, T} <: MeanSpec{T}
An ARMA(p, q) mean specification.
"""

struct ARMA{p, q, T} <: MeanSpec{T}
    coefs::Vector{T}
    function ARMA{p, q, T}(coefs::Vector) where {p, q, T}
        length(coefs) == nparams(ARMA{p, q}) || throw(NumParamError(nparams(ARMA{p, q}), length(coefs)))
        new{p, q, T}(coefs)
    end
end

"""
    ARMA{p, q}(coefs::Vector)
Create an ARMA(p, q) model.
"""
ARMA{p, q}(coefs::Vector{T}) where {p, q, T} = ARMA{p, q, T}(coefs)
nparams(::Type{<:ARMA{p, q}}) where {p, q} = p+q+1
function coefnames(::ARMA{p, q}) where {p, q}
    names = Array{String, 1}(undef, p+q+1)
    names[1] = "c"
    names[2:p+1] .= (i -> "φ"*subscript(i)).([1:p...])
    names[2+p:p+q+1] .= (i -> "θ"*subscript(i)).([1:q...])
    return names
end

const AR{p} = ARMA{p, 0}
const MA{q} = ARMA{0, q}

@inline presample(::ARMA{p, q}) where {p, q} = max(p, q)

Base.@propagate_inbounds @inline function mean(
    at, ht, lht, data, meanspec::ARMA{p, q}, meancoefs::Vector{T}, t
    ) where {p, q, T}
    m = meancoefs[1]
    for i = 1:p
        m += meancoefs[1+i] * data[t-i]
    end
    for i= 1:q
        m += meancoefs[1+p+i] * at[end-i+1]
    end
    return m
end


function constraints(::Type{<:ARMA{p, q}}, ::Type{T})  where {T<:AbstractFloat, p, q}
    lower = [T(-Inf), -ones(T, p+q, 1)...]
    upper = [T(Inf), ones(T, p+q)...]
    return lower, upper
end

function startingvals(::ARMA{p, q, T}, data::Vector{T})  where {p, q, T<:AbstractFloat}
    N = length(data)
    X = Matrix{T}(undef, N-p, p+1)
    X[:, 1] .= T(1)
    for i = 1:p
        X[:, i+1] .= data[p-i+1:N-i]
    end
    phi = X \ data[p+1:end]
    return T[phi..., zeros(T, q)...]
end


Base.@propagate_inbounds @inline function uncond(ms::ARMA{p, q}) where {p, q}
    m = ms.coefs[1]
    p>0 && (m/=(1-sum(ms.coefs[2:p+1])))
    return m
end

################################################################################
#regression
"""
    Regression{k, T} <: MeanSpec{T}
A linear regression as mean specification.
"""
struct Regression{k, T} <: MeanSpec{T}
    coefs::Vector{T}
    X::Matrix{T}
    coefnames::Vector{String}
    function Regression{k, T}(coefs, X; coefnames=(i -> "β"*subscript(i)).([0:(k-1)...])) where {k, T}
        X = X[:, :]
        nparams(Regression{k, T}) == size(X, 2) == length(coefnames) == k || throw(NumParamError(size(X, 2), length(coefs)))
        return new{k, T}(coefs, X, coefnames)
    end
end
"""
    Regression(coefs::Vector, X::Matrix; coefnames=[β₀, β₁, …])
    Regression(X::Matrix; coefnames=[β₀, β₁, …])
    Regression{T}(X::Matrix; coefnames=[β₀, β₁, …])
Create a regression model.
"""
Regression(coefs::Vector{T}, X::MatOrVec{T}; kwargs...) where {T} = Regression{length(coefs), T}(coefs, X; kwargs...)
Regression(coefs::Vector, X::MatOrVec; kwargs...) = (T = float(promote_type(eltype(coefs), eltype(X))); Regression{length(coefs), T}(convert.(T, coefs), convert.(T, X); kwargs...))
Regression{T}(X::MatOrVec; kwargs...) where T = Regression(Vector{T}(undef, size(X, 2)), convert.(T, X); kwargs...)
Regression(X::MatOrVec{T}; kwargs...) where T<:AbstractFloat = Regression{T}(X; kwargs...)
Regression(X::MatOrVec; kwargs...) = Regression(float.(X); kwargs...)
nparams(::Type{Regression{k, T}}) where {k, T} = k
function coefnames(R::Regression{k, T}) where {k, T}
    return R.coefnames
end

@inline presample(::Regression) = 0

Base.@propagate_inbounds @inline function mean(
    at, ht, lht, data, meanspec::Regression{k}, meancoefs::Vector{T}, t
    ) where {k, T}
    t > size(meanspec.X, 1) && error("insufficient number of observations in X (T=$(size(meanspec.X, 1))) to evaluate conditional mean at $t. Consider padding the design matrix. If you are simulating, consider passing `warmup=0`.")
    mean = T(0)
    for i = 1:k
        mean += meancoefs[i] * meanspec.X[t, i]
    end
    return mean
end

function constraints(::Type{<:Regression{k}}, ::Type{T})  where {k, T}
    lower = Vector{T}(undef, k)
    upper = Vector{T}(undef, k)
    fill!(lower, -T(Inf))
    fill!(upper, T(Inf))

    return lower, upper
end

function startingvals(reg::Regression{k, T}, data::Vector{T})  where {k, T<:AbstractFloat}
    N = length(data)
    beta = reg.X[1:N, :] \ data # allow extra entries in X for prediction
end


Base.@propagate_inbounds @inline function uncond(::Regression{k, T}) where {k, T}
    return T(0)
end
