"""
    NGARCH{p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}

Nonlinear GARCH (N-GARCH) of Higgins and Bera (1992); equivalently APARCH
without leverage in the Hansen and Lunde (2005) nesting.

    σ_t^δ = ω + Σ_{i=1}^q α_i |ε_{t-i}|^δ + Σ_{j=1}^p β_j σ_{t-j}^δ

Coefficient vector order is `[ω, β₁, …, βₚ, α₁, …, α_q, δ]`, so
`nparams = p+q+2`. A typical NGARCH(1,1) is `NGARCH{1,1}`. With `δ=2` this
nests [`GARCH{p,q}`](@ref). Implemented as a standalone specification (APARCH
is not on master yet); after that merge it could become an alias for
`APARCH{0,p,q}`.
"""
struct NGARCH{p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
    coefs::Vector{T}
    function NGARCH{p, q, T}(coefs::Vector{T}) where {p, q, T}
        length(coefs) == nparams(NGARCH{p, q})  || throw(NumParamError(nparams(NGARCH{p, q}), length(coefs)))
        new{p, q, T}(coefs)
    end
end

"""
    NGARCH{p, q}(coefs) -> UnivariateVolatilitySpec

Construct an NGARCH specification with the given parameters.

# Example:
```jldoctest
julia> NGARCH{1, 1}([1., .8, .05, 1.5])
NGARCH{1, 1} specification.

────────────────────────────────
               ω   β₁    α₁    δ
────────────────────────────────
Parameters:  1.0  0.8  0.05  1.5
────────────────────────────────
```
"""
NGARCH{p, q}(coefs::Vector{T}) where {p, q, T}  = NGARCH{p, q, T}(coefs)

@inline nparams(::Type{<:NGARCH{p, q}}) where {p, q} = p+q+2
@inline nparams(::Type{<:NGARCH{p, q}}, subset) where {p, q} = isempty(subset) ? 2 : sum(subset) + 2

@inline presample(::Type{<:NGARCH{p, q}}) where {p, q} = max(p, q)

@inline supports_multistep_variance(::Type{<:NGARCH}) = true

Base.@propagate_inbounds @inline function update!(
        ht, lht, zt, at, ::Type{<:NGARCH{p, q}}, garchcoefs,
		current_horizon=1
        ) where {p, q}
    δ = garchcoefs[end]
    σδ = garchcoefs[1]
    halfδ = δ / 2
    @muladd begin
		for i = 1:p
        	σδ = σδ + garchcoefs[i+1]*ht[end-i+1]^halfδ
    	end
    	for i = 1:q
			αi = garchcoefs[i+1+p]
			if i >= current_horizon
        		σδ = σδ + αi*abs(at[end-i+1])^δ
			else
				# multi-step: unknown future residual; use σ^δ (exact if δ=2)
				σδ = σδ + αi*ht[end-i+1]^halfδ
			end
    	end
	end
    mht = σδ > 0 ? σδ^(2 / δ) : σδ
    push!(ht, mht)
    push!(lht, (mht > 0) ? log(mht) : -mht)
    return nothing
end

@inline function uncond(::Type{<:NGARCH{p, q}}, coefs::Vector{T}) where {p, q, T}
    δ = coefs[end]
    den = one(T)
    @inbounds for i = 1:p
        den -= coefs[i+1]
    end
    # E[|Z|^δ] for Z~N(0,1): 2^{δ/2} Γ((δ+1)/2)/√π
    # so δ=2 ⇒ κ=1 and uncond reduces to the GARCH formula ω/(1-Σα-Σβ).
    elogabs = exp(log(T(2)) * (δ / 2)) * gamma((δ + 1) / 2) / sqrt(T(π))
    @inbounds for i = 1:q
        αi = coefs[i+1+p]
        den -= αi * elogabs
    end
    σδ = coefs[1] / den
    h0 = σδ > 0 ? σδ^(2 / δ) : σδ
end

function startingvals(::Type{<:NGARCH{p,q}}, data::Array{T}) where {p, q, T}
    x0 = zeros(T, p+q+2)
    x0[2:p+1] .= T(0.8) / p
    x0[p+2:p+q+1] .= T(0.05) / q
    x0[end] = T(2)
    x0[1] = one(T)
    u = uncond(NGARCH{p, q}, x0)
    h = var(data)
    if u > 0 && isfinite(u)
        x0[1] = (h / u)^(x0[end] / 2)
    else
        x0[1] = h
    end
    return x0
end

function startingvals(TT::Type{<:NGARCH}, data::Array{T} , subset::Tuple) where {T}
	p, q = subsettuple(TT, subsetmask(TT, subset))
	x0 = zeros(T, p+q+2)
    x0[2:p+1] .= T(0.8) / p
    x0[p+2:p+q+1] .= T(0.05) / q
    x0[end] = T(2)
    x0[1] = one(T)
    u = uncond(NGARCH{p, q}, x0)
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

function constraints(::Type{<:NGARCH{p,q}}, ::Type{T}) where {p, q, T}
    lower = zeros(T, p+q+2)
    upper = ones(T, p+q+2)
    upper[1] = T(Inf)
    upper[end] = T(Inf)
    return lower, upper
end

function coefnames(::Type{<:NGARCH{p,q}}) where {p, q}
    names = Array{String, 1}(undef, p+q+2)
    names[1] = "ω"
    names[2:p+1] .= (i -> "β"*subscript(i)).([1:p...])
    names[p+2:p+q+1] .= (i -> "α"*subscript(i)).([1:q...])
    names[end] = "δ"
    return names
end

@inline function subsetmask(VS_large::Union{Type{NGARCH{p, q}}, Type{NGARCH{p, q, T}}}, subs) where {p, q, T}
	ind = falses(nparams(VS_large))
	subset = zeros(Int, 2)
	subset[3-length(subs):end] .= subs
	ind[1] = true
	ps = subset[1]
	qs = subset[2]
	@assert ps <= p
	@assert qs <= q
	ind[2:2+ps-1] .= true
	ind[2+p:2+p+qs-1] .= true
	ind[end] = true # δ is not a lag parameter
	ind
end

@inline function subsettuple(VS_large::Union{Type{NGARCH{p, q}}, Type{NGARCH{p, q, T}}}, subsetmask) where {p, q, T}
	ps = 0
	qs = 0
	@inbounds @simd ivdep for i = 2 : p + 1
		ps += subsetmask[i]
	end
	@inbounds @simd ivdep for i = p + 2 : p + q + 1
		qs += subsetmask[i]
	end
	(ps, qs)
end

# Master `predict` special-cases only TGARCH for multi-step variance.
# This method enables NGARCH multi-step forecasts; unknown future residuals
# use σ^δ inside update! (current_horizon branch). After APARCH lands, this
# can share the generic `supports_multistep_variance` path.
function predict(am::UnivariateARCHModel{T, VS, SD}, what=:volatility, horizon=1; level=0.01) where {T, VS<:NGARCH, SD}
	ht = volatilities(am).^2
	lht = log.(ht)
	zt = residuals(am)
	at = residuals(am, standardized=false)
	themean = T(0)
	if horizon > 1
		if what == :VaR
			error("Predicting VaR more than one period ahead is not implemented. Consider predicting one period ahead and scaling by `sqrt(horizon)`.")
		elseif what == :volatility
			error("Predicting volatility more than one period ahead is not implemented.")
		end
	end
    data = copy(am.data)
	for current_horizon = (1 : horizon)
		t = length(am.data) + current_horizon
		if what == :return || what == :VaR
			themean = mean(at, ht, lht, data, am.meanspec, am.meanspec.coefs, t)
		end
		update!(ht, lht, zt, at, VS, am.spec.coefs, current_horizon)
		push!(zt, 0.)
		push!(at, 0.)
        push!(data, themean)
	end
	if what == :return
		return themean
	elseif what == :volatility
		return sqrt(ht[end])
	elseif what == :variance
		return ht[end]
	elseif what == :VaR
		return -themean - sqrt(ht[end]) * quantile(am.dist, level)
	else error("Prediction target $what unknown.")
	end
end
