struct DCC{p, q, VS<:VolatilitySpec, SD<:StandardizedDistribution, MS<:MeanSpec, T<:AbstractFloat, d} <: MultivariateVolatilitySpec{T, d}
    Σ::Matrix{T}
    coefs::Vector{T}
    univariatespecs::Vector{UnivariateARCHModel{T, VS, SD, MS}}
    function DCC{p, q, VS, SD, MS, T, d}(Σ::Array{T}, coefs::Vector{T}, univariatespecs:: Array{UnivariateARCHModel{T, VS, SD, MS}}) where {p, q, T, VS<:VolatilitySpec, SD<:StandardizedDistribution, MS<:MeanSpec, d}
        length(coefs) == nparams(DCC{p, q})  || throw(NumParamError(nparams(DCC{p, q}), length(coefs)))
        @assert d == length(univariatespecs)
        new{p, q, VS, SD, MS, T, d}(Σ, coefs, univariatespecs)
    end
end
DCC{p, q}(Σ::Matrix{T}, coefs::Vector{T}, univariatespecs::Array{UnivariateARCHModel{T, VS, SD, MS}}) where {p, q, T, VS<:VolatilitySpec{T}, SD<:StandardizedDistribution{T}, MS<:MeanSpec{T} } = DCC{p, q, VS, SD, MS, T, length(univariatespecs)}(Σ, coefs, univariatespecs)

nparams(::Type{DCC{p, q}}) where {p, q} = p+q

fit(::Type{<:DCC}, data) = fit(DCC{1, 1}, data)

fit(DCCspec::Type{<:DCC{p, q}}, data) where {p, q} = fit(DCC{p, q, GARCH{1, 1}}, data)

function fit(DCCspec::Type{<:DCC{p, q, VS}}, data::Matrix{T}) where {p, q, VS<: VolatilitySpec, T, d}
    n, dim = size(data)
    resids = similar(data)
    m = fit(VS, data[:, 1])
    resids[:, 1] = residuals(m)
    univariatespecs = [m]
    Threads.@threads for i = 2:dim
        m = fit(VS, data[:, i])
        push!(univariatespecs, m)
        resids[:, i] = residuals(m)
    end
    Σ = analytical_shrinkage(resids)
    D = sqrt(Diagonal(Σ))
    iD = inv(D)
    R = iD * Σ * iD

    res = optimize(x->-LL2step(x, R, resids, p, q), [.05, .9], BFGS(), autodiff=:forward)
    return DCC{p, q}(Σ, Optim.minimizer(res), univariatespecs)
end

function LL2step(coef::Array{T}, R, resids, p, q) where {T}
    all([0, 0] .< coef .< [1, 1]) || return T(-Inf)
    n, dims = size(resids)
    LL = zero(eltype(coef))
    a = coef[1]
    b = coef[2]
    abs(a+b)>1 && return T(-Inf)
    e = resids[1, :]
    Rt = copy(R)
    for t=1:n
        if t > max(p, q)
            Rt = R.* (1-a-b) .+ a .* e*e' .+ b .* Rt
            RD5 = inv(sqrt(Diagonal(Rt)))
            Rt = RD5 * Rt * RD5
            Rt = (Rt + Rt')/2
        end
        e .= resids[t, :]
        Ci = inv(cholesky(Rt, check=false).L)
        u = Ci*e
        LL -= (dot(u, u) - dot(e, e))/2-logdet(Ci)
    end
    LL
end
