"""
    EGARCH{o, p, q, T<:AbstractFloat} <: VolatilitySpec{T}
"""
struct EGARCH{o, p, q, T<:AbstractFloat} <: VolatilitySpec{T}
    coefs::Vector{T}
    function EGARCH{o, p, q, T}(coefs::Vector{T}) where {o, p, q, T}
        length(coefs) == nparams(EGARCH{o, p, q})  || throw(NumParamError(nparams(EGARCH{o, p, q}), length(coefs)))
        new{o, p, q, T}(coefs)
    end
end

"""
    EGARCH{o, p, q}(coefs) -> VolatilitySpec

Construct an EGARCH specification with the given parameters.

# Example:
```jldoctest
julia> EGARCH{1, 1, 1}([-0.1, .1, .9, .04])
EGARCH{1,1,1} specification.

                ω  γ₁  β₁   α₁
Parameters:  -0.1 0.1 0.9 0.04
```
"""
EGARCH{o, p, q}(coefs::Vector{T}) where {o, p, q, T}  = EGARCH{o, p, q, T}(coefs)

@inline nparams(::Type{<:EGARCH{o, p, q}}) where {o, p, q} = o+p+q+1

@inline presample(::Type{<:EGARCH{o, p, q}}) where {o, p, q} = max(o, p, q)

Base.@propagate_inbounds @inline function update!(
            ht, lht, zt, at, ::Type{<:EGARCH{o, p ,q}}, meanspec::MeanSpec,
            data, garchcoefs, meancoefs
            ) where {o, p, q}
    mlht = garchcoefs[1]
    for i = 1:o
        mlht += garchcoefs[i+1]*zt[end-i+1]
    end
    for i = 1:p
        mlht += garchcoefs[i+1+o]*lht[end-i+1]
    end
    for i = 1:q
        mlht += garchcoefs[i+1+o+p]*(abs(zt[end-i+1]) - sqrt2invpi)
    end
    push!(lht, mlht)
    push!(ht, exp(mlht))
    return nothing
end

@inline function uncond(::Type{<:EGARCH{o, p, q}}, coefs::Vector{T}) where {o, p, q, T}
    eg = one(T)
    for i=1:max(o, q)
        γ = (i<=o ? coefs[1+i] : zero(T))
        α = (i<=q ? coefs[o+p+1+i] : zero(T))
        eg *= exp(-α*sqrt2invpi) * (exp(.5*(γ+α)^2)*normcdf(γ+α) + exp(.5*(γ-α)^2)*normcdf(α-γ))
    end
    h0 = (exp(coefs[1])*eg)^(1/(1-sum(coefs[o+2:o+p+1])))
end

function startingvals(spec::Type{<:EGARCH{o, p, q}}, data::Array{T}) where {o, p, q, T}
    x0 = zeros(T, o+p+q+1)
    x0[1]=1
    x0[2:o+1] .= 0
    x0[o+2:o+p+1] .= 0.9/p
    x0[o+p+2:end] .= 0.05/q
    x0[1] = var(data)/uncond(spec, x0)
    return x0
end

function constraints(::Type{<:EGARCH{o, p,q}}, ::Type{T}) where {o, p, q, T}
    lower = zeros(T, o+p+q+1)
    upper = zeros(T, o+p+q+1)
    lower .=  T(-Inf)
    upper .= T(Inf)
    lower[1] = T(-Inf)
    lower[o+2:o+p+1] .= zero(T)
    upper[o+2:o+p+1] .= one(T)
    return lower, upper
end

function coefnames(::Type{<:EGARCH{o, p, q}}) where {o, p, q}
    names = Array{String, 1}(undef, o+p+q+1)
    names[1] = "ω"
    names[2:o+1] .= (i -> "γ"*subscript(i)).([1:o...])
    names[o+2:o+p+1] .= (i -> "β"*subscript(i)).([1:p...])
    names[o+p+2:o+p+q+1] .= (i -> "α"*subscript(i)).([1:q...])
    return names
end
