export TGARCH
export GARCH
export ARCH


"""
    TGARCH{o, p, q, T<:AbstractFloat} <: VolatilitySpec{T}
"""
struct TGARCH{o, p, q, T<:AbstractFloat} <: VolatilitySpec{T}
    coefs::Vector{T}
    function TGARCH{o, p, q, T}(coefs::Vector{T}) where {o, p, q, T}
        length(coefs) == nparams(TGARCH{o, p, q})  || throw(NumParamError(nparams(TGARCH{o, p, q}), length(coefs)))
        new{o, p, q, T}(coefs)
    end
end

"""
    TGARCH{o, p, q}(coefs) -> VolatilitySpec
Construct a TGARCH specification with the given parameters.

# Example:
```jldoctest
julia> TGARCH{1, 1, 1}([1., .04, .9, .01])
TGARCH{1,1,1} specification.

               ω   γ₁  β₁   α₁
Parameters:  1.0 0.04 0.9 0.01
```
"""
TGARCH{o, p, q}(coefs::Vector{T}) where {o, p, q, T}  = TGARCH{o, p, q, T}(coefs)

"""
    GARCH{p, q, T<:AbstractFloat} <: VolatilitySpec{T}
---
    GARCH{p, q}(coefs) -> VolatilitySpec

Construct a GARCH specification with the given parameters.

# Example:
```jldoctest
julia> GARCH{2, 1}([1., .3, .4, .05 ])
TGARCH{0,2,1} specification.

               ω  β₁  β₂   α₁
Parameters:  1.0 0.3 0.4 0.05
```

"""
const GARCH = TGARCH{0}

"""
    ARCH{q, T<:AbstractFloat} <: VolatilitySpec{T}
---
    ARCH{q}(coefs) -> VolatilitySpec

Construct an ARCH specification with the given parameters.

# Example:
```jldoctest
julia> ARCH{2}([1., .3, .4])
TGARCH{0,0,2} specification.

               ω  α₁  α₂
Parameters:  1.0 0.3 0.4
```
"""
const ARCH = GARCH{0}

@inline nparams(::Type{<:TGARCH{o, p, q}}) where {o, p, q} = o+p+q+1

@inline presample(::Type{<:TGARCH{o, p, q}}) where {o, p, q} = max(o, p, q)

Base.@propagate_inbounds @inline function update!(
        ht, lht, zt, at, ::Type{<:TGARCH{o, p, q}}, meanspec::MeanSpec,
        data, garchcoefs, meancoefs
        ) where {o, p, q}
    mht = garchcoefs[1]
    for i = 1:o
        mht += garchcoefs[i+1]*min(at[end-i+1], 0)^2
    end
    for i = 1:p
        mht += garchcoefs[i+1+o]*ht[end-i+1]
    end
    for i = 1:q
        mht += garchcoefs[i+1+o+p]*(at[end-i+1])^2
    end
    push!(ht, mht)
    push!(lht, (mht > 0) ? log(mht) : -mht)
    return nothing
end

@inline function uncond(::Type{<:TGARCH{o, p, q}}, coefs::Vector{T}) where {o, p, q, T}
    den=one(T)
    for i = 1:o
        den -= coefs[i+1]/2
    end
    for i = o+1:o+p+q
        den -= coefs[i+1]
    end
    h0 = coefs[1]/den
end

function startingvals(::Type{<:TGARCH{o,p,q}}, data::Array{T}) where {o, p, q, T}
    x0 = zeros(T, o+p+q+1)
    x0[2:o+1] .= 0.04/o
    x0[o+2:o+p+1] .= 0.9/p
    x0[o+p+2:end] .= o>0 ? 0.01/q : 0.05/q
    x0[1] = var(data)*(one(T)-sum(x0[2:o+1])/2-sum(x0[o+2:end]))
    return x0
end

function constraints(::Type{<:TGARCH{o,p,q}}, ::Type{T}) where {o,p, q, T}
    lower = zeros(T, o+p+q+1)
    upper = ones(T, o+p+q+1)
    upper[2:o+1] .= ones(T, o)/2
    upper[1] = T(Inf)
    return lower, upper
end

function coefnames(::Type{<:TGARCH{o,p,q}}) where {o,p, q}
    names = Array{String, 1}(undef, o+p+q+1)
    names[1] = "ω"
    names[2:o+1] .= (i -> "γ"*subscript(i)).([1:o...])
    names[2+o:o+p+1] .= (i -> "β"*subscript(i)).([1:p...])
    names[o+p+2:o+p+q+1] .= (i -> "α"*subscript(i)).([1:q...])
    return names
end
