"""
    TGARCH{o, p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
"""
struct TGARCH{o, p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
    coefs::Vector{T}
    function TGARCH{o, p, q, T}(coefs::Vector{T}) where {o, p, q, T}
        length(coefs) == nparams(TGARCH{o, p, q})  || throw(NumParamError(nparams(TGARCH{o, p, q}), length(coefs)))
        new{o, p, q, T}(coefs)
    end
end

"""
    TGARCH{o, p, q}(coefs) -> UnivariateVolatilitySpec
Construct a TGARCH specification with the given parameters.

# Example:
```jldoctest
julia> TGARCH{1, 1, 1}([1., .04, .9, .01])
TGARCH{1,1,1} specification.

─────────────────────────────────
               ω    γ₁   β₁    α₁
─────────────────────────────────
Parameters:  1.0  0.04  0.9  0.01
─────────────────────────────────
```
"""
TGARCH{o, p, q}(coefs::Vector{T}) where {o, p, q, T}  = TGARCH{o, p, q, T}(coefs)

"""
    GARCH{p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
---
    GARCH{p, q}(coefs) -> UnivariateVolatilitySpec

Construct a GARCH specification with the given parameters.

# Example:
```jldoctest
julia> GARCH{2, 1}([1., .3, .4, .05 ])
TGARCH{0,2,1} specification.

────────────────────────────────
               ω   β₁   β₂    α₁
────────────────────────────────
Parameters:  1.0  0.3  0.4  0.05
────────────────────────────────
```

"""
const GARCH = TGARCH{0}

"""
    ARCH{q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
---
    ARCH{q}(coefs) -> UnivariateVolatilitySpec

Construct an ARCH specification with the given parameters.

# Example:
```jldoctest
julia> ARCH{2}([1., .3, .4])
TGARCH{0,0,2} specification.

──────────────────────────
               ω   α₁   α₂
──────────────────────────
Parameters:  1.0  0.3  0.4
──────────────────────────
```
"""
const ARCH = GARCH{0}

@inline nparams(::Type{<:TGARCH{o, p, q}}) where {o, p, q} = o+p+q+1
@inline nparams(::Type{<:TGARCH{o, p, q}}, subset) where {o, p, q} = isempty(subset) ? 1 : sum(subset) + 1

@inline presample(::Type{<:TGARCH{o, p, q}}) where {o, p, q} = max(o, p, q)


Base.@propagate_inbounds @inline function update!(
        ht, lht, zt, at, ::Type{<:TGARCH{o, p, q}}, garchcoefs,
		current_horizon=1
        ) where {o, p, q}
    mht = garchcoefs[1]
    @muladd begin
		for i = 1:o
        	mht = mht + garchcoefs[i+1]*min(at[end-i+1], 0)^2
    	end
    	for i = 1:p
        	mht = mht + garchcoefs[i+1+o]*ht[end-i+1]
    	end
    	for i = 1:q
<<<<<<< HEAD
        	mht = mht + garchcoefs[i+1+o+p]*(at[end-i+1])^2
=======
			if i >= current_horizon
        		mht = mht + garchcoefs[i+1+o+p]*(at[end-i+1])^2
			else
				mht = mht + garchcoefs[i+1+o+p]*ht[end-i+1]
			end
>>>>>>> 1b567383213740fbf46de5df3b4d5280b8a63298
    	end
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

function startingvals(TT::Type{<:TGARCH}, data::Array{T} , subset::Tuple) where {T}
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

@inline function subsetmask(VS_large::Union{Type{TGARCH{o, p, q}}, Type{TGARCH{o, p, q, T}}}, subs) where {o, p, q, T}
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

@inline function subsettuple(VS_large::Union{Type{TGARCH{o, p, q}}, Type{TGARCH{o, p, q, T}}}, subsetmask) where {o, p, q, T}
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
