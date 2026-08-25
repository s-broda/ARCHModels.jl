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
