"""
    EGARCH{o, p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
"""
struct EGARCH{o, p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
    coefs::Vector{T}
    function EGARCH{o, p, q, T}(coefs::Vector{T}) where {o, p, q, T}
        length(coefs) == nparams(EGARCH{o, p, q})  || throw(NumParamError(nparams(EGARCH{o, p, q}), length(coefs)))
        new{o, p, q, T}(coefs)
    end
end

"""
    EGARCH{o, p, q}(coefs) -> UnivariateVolatilitySpec

Construct an EGARCH specification with the given parameters.

# Example:
```jldoctest
julia> EGARCH{1, 1, 1}([-0.1, .1, .9, .04])
EGARCH{1,1,1} specification.

─────────────────────────────────
                ω   γ₁   β₁    α₁
─────────────────────────────────
Parameters:  -0.1  0.1  0.9  0.04
─────────────────────────────────
```
"""
EGARCH{o, p, q}(coefs::Vector{T}) where {o, p, q, T}  = EGARCH{o, p, q, T}(coefs)

@inline nparams(::Type{<:EGARCH{o, p, q}}) where {o, p, q} = o+p+q+1
@inline nparams(::Type{<:EGARCH{o, p, q}}, subset) where {o, p, q} = isempty(subset) ? 1 : sum(subset) + 1

@inline presample(::Type{<:EGARCH{o, p, q}}) where {o, p, q} = max(o, p, q)

Base.@propagate_inbounds @inline function update!(
            ht, lht, zt, at, ::Type{<:EGARCH{o, p ,q}}, garchcoefs
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

function startingvals(TT::Type{<:EGARCH}, data::Array{T} , subset::Tuple) where {T}
	o, p, q = subsettuple(TT, subsetmask(TT, subset)) # defend against (p, q) instead of (o, p, q)
	x0 = zeros(T, o+p+q+1)
    x0[2:o+1] .= 0.04/o
    x0[o+2:o+p+1] .= 0.9/p
    x0[o+p+2:end] .= o>0 ? 0.01/q : 0.05/q
    x0[1] = var(data)*(one(T)-sum(x0[2:o+1])/2-sum(x0[o+2:end]))
	mask = subsetmask(TT, subset)
	x0long = zeros(T, length(mask))
	x0long[mask] .= x0
    return x0long
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

@inline function subsetmask(VS_large::Union{Type{EGARCH{o, p, q}}, Type{EGARCH{o, p, q, T}}}, subs) where {o, p, q, T}
	ind = falses(nparams(VS_large))
	subset = zeros(Int, 3)
	subset[4-length(subs):end] .= subs
	ind[1] = true
	os = subset[1]
	ps = subset[2]
	qs = subset[3]
	@assert os <= o
	@assert ps <= p
	@assert qs <= q
	ind[2:2+os-1] .= true
	ind[2+o:2+o+ps-1] .= true
	ind[2+o+p:2+o+p+qs-1] .= true
	ind
end

@inline function subsettuple(VS_large::Union{Type{EGARCH{o, p, q}}, Type{EGARCH{o, p, q, T}}}, subsetmask) where {o, p, q, T}
	os = 0
	ps = 0
	qs = 0
	@inbounds @simd ivdep for i = 2 : o + 1
		os += subsetmask[i]
	end
	@inbounds @simd ivdep for i = o + 2 : o + p + 1
		ps += subsetmask[i]
	end
	@inbounds @simd ivdep for i = o + p + 2 : o + p + q + 1
		qs += subsetmask[i]
	end
	(os, ps, qs)
end
