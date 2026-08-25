"""
    APARCH{o, p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}

Asymmetric Power ARCH of Ding, Granger and Engle (1993).

Type parameters follow the `TGARCH{o,p,q}` convention:
- `o`: leverage / asymmetry lags (γ)
- `p`: GARCH lags (β)
- `q`: ARCH lags (α)

together with intercept ω and power δ. Coefficient vector order is
`[ω, γ₁, …, γₒ, β₁, …, βₚ, α₁, …, α_q, δ]`.

The recursion (rugarch / Ding et al. 1993) is

    σ_t^δ = ω + Σ_{i=1}^q α_i (|ε_{t-i}| - γ_i ε_{t-i})^δ + Σ_{j=1}^p β_j σ_{t-j}^δ

with δ>0 and |γ|<1. When `i>o`, γ_i is treated as 0, so `APARCH{0,p,q}` is
APARCH without leverage (NGARCH in the Hansen & Lunde 2005 nesting).
`APARCH{0,p,q}` with δ=2 further nests `GARCH{p,q}`.
"""
struct APARCH{o, p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
    coefs::Vector{T}
    function APARCH{o, p, q, T}(coefs::Vector{T}) where {o, p, q, T}
        length(coefs) == nparams(APARCH{o, p, q})  || throw(NumParamError(nparams(APARCH{o, p, q}), length(coefs)))
        new{o, p, q, T}(coefs)
    end
end

"""
    APARCH{o, p, q}(coefs) -> UnivariateVolatilitySpec

Construct an APARCH specification with the given parameters.

# Example:
```jldoctest
julia> APARCH{1, 1, 1}([1., .1, .8, .05, 1.5])
APARCH{1, 1, 1} specification.

─────────────────────────────────────
               ω   γ₁   β₁    α₁    δ
─────────────────────────────────────
Parameters:  1.0  0.1  0.8  0.05  1.5
─────────────────────────────────────
```
"""
APARCH{o, p, q}(coefs::Vector{T}) where {o, p, q, T}  = APARCH{o, p, q, T}(coefs)

@inline nparams(::Type{<:APARCH{o, p, q}}) where {o, p, q} = o+p+q+2
@inline nparams(::Type{<:APARCH{o, p, q}}, subset) where {o, p, q} = isempty(subset) ? 2 : sum(subset) + 2

@inline presample(::Type{<:APARCH{o, p, q}}) where {o, p, q} = max(o, p, q)

Base.@propagate_inbounds @inline function update!(
        ht, lht, zt, at, ::Type{<:APARCH{o, p, q}}, garchcoefs,
		current_horizon=1
        ) where {o, p, q}
    δ = garchcoefs[end]
    σδ = garchcoefs[1]
    halfδ = δ / 2
    @muladd begin
		for i = 1:p
        	σδ = σδ + garchcoefs[i+1+o]*ht[end-i+1]^halfδ
    	end
    	for i = 1:q
			αi = garchcoefs[i+1+o+p]
			if i >= current_horizon
				γi = (i <= o ? garchcoefs[i+1] : zero(eltype(garchcoefs)))
				a = at[end-i+1]
				c = abs(a) - γi*a
        		σδ = σδ + αi*abs(c)^δ
			else
				# multi-step: unknown future residual; use σ^δ (exact only if γ=0, δ=2)
				σδ = σδ + αi*ht[end-i+1]^halfδ
			end
    	end
	end
    mht = σδ > 0 ? σδ^(2 / δ) : σδ
    push!(ht, mht)
    push!(lht, (mht > 0) ? log(mht) : -mht)
    return nothing
end

@inline function uncond(::Type{<:APARCH{o, p, q}}, coefs::Vector{T}) where {o, p, q, T}
    δ = coefs[end]
    den = one(T)
    @inbounds for i = 1:p
        den -= coefs[i+1+o]
    end
    # E[(|Z|-γZ)^δ] for Z~N(0,1): ½[(1+γ)^δ+(1-γ)^δ] × 2^{δ/2} Γ((δ+1)/2)/√π
    # so δ=2, γ=0 ⇒ κ=1 and uncond reduces to the GARCH formula ω/(1-Σα-Σβ).
    elogabs = exp(log(T(2)) * (δ / 2)) * gamma((δ + 1) / 2) / sqrt(T(π))
    @inbounds for i = 1:q
        γi = i <= o ? coefs[i+1] : zero(T)
        αi = coefs[i+1+o+p]
        κ = ((1 + γi)^δ + (1 - γi)^δ) / 2 * elogabs
        den -= αi * κ
    end
    σδ = coefs[1] / den
    h0 = σδ > 0 ? σδ^(2 / δ) : σδ
end

function startingvals(::Type{<:APARCH{o,p,q}}, data::Array{T}) where {o, p, q, T}
    x0 = zeros(T, o+p+q+2)
    x0[2:o+1] .= T(0.1)
    x0[o+2:o+p+1] .= T(0.8) / p
    x0[o+p+2:o+p+q+1] .= T(0.05) / q
    x0[end] = T(2)
    x0[1] = one(T)
    u = uncond(APARCH{o, p, q}, x0)
    h = var(data)
    if u > 0 && isfinite(u)
        x0[1] = (h / u)^(x0[end] / 2)
    else
        x0[1] = h
    end
    return x0
end

function startingvals(TT::Type{<:APARCH}, data::Array{T} , subset::Tuple) where {T}
	o, p, q = subsettuple(TT, subsetmask(TT, subset)) # defend against (p, q) instead of (o, p, q)
	x0 = zeros(T, o+p+q+2)
    x0[2:o+1] .= T(0.1)
    x0[o+2:o+p+1] .= T(0.8) / p
    x0[o+p+2:o+p+q+1] .= T(0.05) / q
    x0[end] = T(2)
    x0[1] = one(T)
    u = uncond(APARCH{o, p, q}, x0)
    h = var(data)
    if u > 0 && isfinite(u)
        x0[1] = (h / u)^(x0[end] / 2)
    else
        x0[1] = h
    end
	mask = subsetmask(TT, subset)
	x0long = zeros(T, length(mask))
	x0long[mask] .= x0
    return x0long
end

function constraints(::Type{<:APARCH{o,p,q}}, ::Type{T}) where {o,p, q, T}
    lower = zeros(T, o+p+q+2)
    upper = ones(T, o+p+q+2)
    lower[2:o+1] .= -one(T)
    upper[2:o+1] .= one(T)
    upper[1] = T(Inf)
    upper[end] = T(Inf)
    return lower, upper
end

function coefnames(::Type{<:APARCH{o,p,q}}) where {o,p, q}
    names = Array{String, 1}(undef, o+p+q+2)
    names[1] = "ω"
    names[2:o+1] .= (i -> "γ"*subscript(i)).([1:o...])
    names[o+2:o+p+1] .= (i -> "β"*subscript(i)).([1:p...])
    names[o+p+2:o+p+q+1] .= (i -> "α"*subscript(i)).([1:q...])
    names[end] = "δ"
    return names
end

@inline function subsetmask(VS_large::Union{Type{APARCH{o, p, q}}, Type{APARCH{o, p, q, T}}}, subs) where {o, p, q, T}
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
	ind[end] = true # δ is not a lag parameter
	ind
end

@inline function subsettuple(VS_large::Union{Type{APARCH{o, p, q}}, Type{APARCH{o, p, q, T}}}, subsetmask) where {o, p, q, T}
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
