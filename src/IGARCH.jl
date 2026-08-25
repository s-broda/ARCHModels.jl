"""
    IGARCH{p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}

Integrated GARCH of Engle and Bollerslev (1986). Same recursion as
[`GARCH{p,q}`](@ref), with the equality constraint that the ARCH and GARCH
lags sum to one:

    σ²_t = ω + Σ_{i=1}^q α_i a_{t-i}² + Σ_{j=1}^p β_j σ²_{t-j},   Σα + Σβ = 1.

The unconditional variance is infinite, and multi-step variance forecasts do
not mean-revert. Requires `p ≥ 1` (a GARCH lag to hang the unit root on).
Following rugarch, the last GARCH coefficient is implied:

    β_p = 1 - Σα - Σ_{j<p} β_j.

Free coefficients are stored as `(ω, β_1, …, β_{p-1}, α_1, …, α_q)`,
so `nparams = p+q`. The implied `β_p` is reconstructed inside `update!`
and is not an estimable parameter (the optimizer only has box constraints).
"""
struct IGARCH{p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
    coefs::Vector{T}
    function IGARCH{p, q, T}(coefs::Vector{T}) where {p, q, T}
        p >= 1 || throw(ArgumentError("IGARCH requires p ≥ 1 (need a GARCH lag to hang the unit root on)."))
        length(coefs) == nparams(IGARCH{p, q})  || throw(NumParamError(nparams(IGARCH{p, q}), length(coefs)))
        new{p, q, T}(coefs)
    end
end

"""
    IGARCH{p, q}(coefs) -> UnivariateVolatilitySpec

Construct an IGARCH specification with the given parameters. `coefs` holds the
free coefficients `(ω, β₁, …, β_{p-1}, α₁, …, α_q)`; the last GARCH
coefficient is implied as `β_p = 1 - Σα - Σ_{j<p} β_j`.

# Example:
```jldoctest
julia> IGARCH{1, 1}([.1, .05])
IGARCH{1, 1} specification.

────────────────────────────────────
               ω    α₁  β₁ (implied)
────────────────────────────────────
Parameters:  0.1  0.05          0.95
────────────────────────────────────
```
"""
IGARCH{p, q}(coefs::Vector{T}) where {p, q, T}  = IGARCH{p, q, T}(coefs)

@inline nparams(::Type{<:IGARCH{p, q}}) where {p, q} = p+q
@inline nparams(::Type{<:IGARCH{p, q}}, subset) where {p, q} = isempty(subset) ? 1 : sum(subset)

@inline presample(::Type{<:IGARCH{p, q}}) where {p, q} = max(p, q)

"""
    implied_beta(::Type{<:IGARCH{p,q}}, coefs)

Implied last GARCH coefficient `β_p = 1 - Σα - Σ_{j<p} β`.
"""
@inline function implied_beta(::Type{<:IGARCH{p, q}}, coefs) where {p, q}
    s = zero(eltype(coefs))
    @inbounds for i = 2:p+q
        s += coefs[i]
    end
    return one(eltype(coefs)) - s
end
implied_beta(spec::IGARCH) = implied_beta(typeof(spec), spec.coefs)

Base.@propagate_inbounds @inline function update!(
        ht, lht, zt, at, ::Type{<:IGARCH{p, q}}, garchcoefs,
		current_horizon=1
        ) where {p, q}
    mht = garchcoefs[1]
    βp = one(eltype(garchcoefs))
    @muladd begin
		for i = 1:p-1
        	βi = garchcoefs[i+1]
        	βp = βp - βi
        	mht = mht + βi*ht[end-i+1]
    	end
    	for i = 1:q
			αi = garchcoefs[p+i]
			βp = βp - αi
			if i >= current_horizon
        		mht = mht + αi*(at[end-i+1])^2
			else
				mht = mht + αi*ht[end-i+1]
			end
    	end
    	mht = mht + βp*ht[end-p+1]
	end
    # negative implied β_p is infeasible; existing machinery bails on ht < 0
    βp >= 0 || (mht = βp)
    push!(ht, mht)
    push!(lht, (mht > 0) ? log(mht) : -mht)
    return nothing
end

@inline function uncond(::Type{<:IGARCH{p, q}}, coefs::Vector{T}) where {p, q, T}
    T(Inf)
end

@inline supports_multistep_variance(::Type{<:IGARCH}) = true

function startingvals(::Type{<:IGARCH{p, q}}, data::Array{T}) where {p, q, T}
    x0 = zeros(T, p+q)
    # free β₁,…,β_{p-1}; implied β_p absorbs leftover persistence
    if p > 1
        x0[2:p] .= T(0.9) / p
    end
    if q > 0
        x0[p+1:p+q] .= T(0.05) / q
    end
    # do not use var*(1-Σ) which is 0 under the unit-root constraint
    x0[1] = var(data) / T(length(data))
    x0[1] = x0[1] > 0 ? x0[1] : eps(T)
    return x0
end

function startingvals(TT::Type{<:IGARCH}, data::Array{T}, subset::Tuple) where {T}
	p, q = subsettuple(TT, subsetmask(TT, subset))
	x0 = startingvals(IGARCH{p, q}, data)
	mask = subsetmask(TT, subset)
	x0long = zeros(T, length(mask))
	x0long[mask] .= x0
    return x0long
end

function constraints(::Type{<:IGARCH{p, q}}, ::Type{T}) where {p, q, T}
    lower = zeros(T, p+q)
    upper = ones(T, p+q)
    upper[1] = T(Inf)
    return lower, upper
end

function coefnames(::Type{<:IGARCH{p, q}}) where {p, q}
    names = Array{String, 1}(undef, p+q)
    names[1] = "ω"
    names[2:p] .= (i -> "β"*subscript(i)).([1:p-1...])
    names[p+1:p+q] .= (i -> "α"*subscript(i)).([1:q...])
    return names
end

function show(io::IO, ::MIME"text/plain", spec::IGARCH{p, q}) where {p, q}
    println(io, modname(typeof(spec)), " specification.\n")
    βp = implied_beta(spec)
    names = vcat(coefnames(typeof(spec)), "β"*subscript(p)*" (implied)")
    vals = vcat(spec.coefs, βp)
    show(io, "text/plain", CoefTable(vals, names, ["Parameters:"]))
end

@inline function subsetmask(VS_large::Union{Type{IGARCH{p, q}}, Type{IGARCH{p, q, T}}}, subs) where {p, q, T}
	# Free layout: [ω, β₁, …, β_{p-1}, α₁, …, α_q]. The implied β_p of the
	# *large* type is not a stored coefficient, so this mask cannot implement
	# “last *active* β is implied”. selectmodel therefore uses fitsubset, which
	# fits the true IGARCH{p,q} type rather than a zeroed large model.
	ind = falses(nparams(VS_large))
	subset = zeros(Int, 2)
	subset[3-length(subs):end] .= subs
	ind[1] = true
	ps = subset[1]
	qs = subset[2]
	@assert ps <= p
	@assert qs <= q
	ps >= 1 || throw(ArgumentError("IGARCH subsets require p ≥ 1."))
	# first ps-1 of the p-1 free β slots
	ind[2:ps] .= true
	ind[p+1:p+qs] .= true
	ind
end

@inline function subsettuple(VS_large::Union{Type{IGARCH{p, q}}, Type{IGARCH{p, q, T}}}, subsetmask) where {p, q, T}
	# +1 for the implied last GARCH lag
	ps = 1
	qs = 0
	@inbounds @simd ivdep for i = 2:p
		ps += subsetmask[i]
	end
	@inbounds @simd ivdep for i = p+1:p+q
		qs += subsetmask[i]
	end
	(ps, qs)
end

# The generic selectmodel/fitsubset encoding zeros unused lags of a large type
# (e.g. IGARCH{3,3}). That would hang leftover persistence on σ_{t-3}, so a
# subset (1,1) would not be IGARCH(1,1). Fit the true IGARCH{p,q} instead.
function fitsubset(::Type{<:IGARCH}, data::Vector{T}, maxlags::Int, subset::Tuple; dist::Type{SD}=StdNormal{T},
             meanspec::Union{MS, Type{MS}}=Intercept{T}(T[0]), algorithm=BFGS(),
             autodiff=:forward, kwargs...
             ) where {SD<:StandardizedDistribution, MS<:MeanSpec, T<:AbstractFloat}
	length(subset) == 2 || throw(ArgumentError("IGARCH subset must be a (p, q) tuple."))
	p, q = subset
	p >= 1 || throw(ArgumentError("IGARCH requires p ≥ 1 (need a GARCH lag to hang the unit root on)."))
	am = fit(IGARCH{p, q}, data; dist=dist, meanspec=meanspec, algorithm=algorithm, autodiff=autodiff, kwargs...)
	UnivariateSubsetARCHModel(am.spec, am.data; dist=am.dist, meanspec=am.meanspec, fitted=true, subset=subset)
end
