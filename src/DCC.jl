struct DCC{p, q, VS<:VolatilitySpec, SD<:StandardizedDistribution, MS<:MeanSpec, T<:AbstractFloat, d} <: MultivariateVolatilitySpec{T, d}
    R::Matrix{T}
    coefs::Vector{T}
    univariatespecs::Vector{UnivariateARCHModel{T, VS, SD, MS}}
    function DCC{p, q, VS, SD, MS, T, d}(R::Array{T}, coefs::Vector{T}, univariatespecs:: Array{UnivariateARCHModel{T, VS, SD, MS}}) where {p, q, T, VS<:VolatilitySpec, SD<:StandardizedDistribution, MS<:MeanSpec, d}
        length(coefs) == nparams(DCC{p, q})  || throw(NumParamError(nparams(DCC{p, q}), length(coefs)))
        @assert d == length(univariatespecs)
        new{p, q, VS, SD, MS, T, d}(R, coefs, univariatespecs)
    end
end
DCC{p, q}(R::Matrix{T}, coefs::Vector{T}, univariatespecs::Array{UnivariateARCHModel{T, VS, SD, MS}}) where {p, q, T, VS<:VolatilitySpec{T}, SD<:StandardizedDistribution{T}, MS<:MeanSpec{T} } = DCC{p, q, VS, SD, MS, T, length(univariatespecs)}(R, coefs, univariatespecs)

nparams(::Type{DCC{p, q}}) where {p, q} = p+q

fit(::Type{<:DCC}, data) = fit(DCC{1, 1}, data)

fit(DCCspec::Type{<:DCC{p, q}}, data) where {p, q} = fit(DCC{p, q, GARCH{1, 1}}, data)

function fit(DCCspec::Type{<:DCC{p, q, VS}}, data::Matrix{T}) where {p, q, VS<: VolatilitySpec, T, d}
    n, dim = size(data)
    resids = similar(data)
    m = fit(VS, data[:, 1], meanspec=NoIntercept())
    resids[:, 1] = residuals(m)
    univariatespecs = Vector{typeof(m)}(undef, dim)
    univariatespecs[1] = m
    Threads.@threads for i = 2:dim
        m = fit(VS, data[:, i], meanspec=NoIntercept())
        univariatespecs[i] = m
        resids[:, i] = residuals(m)
    end
    #Σ = analytical_shrinkage(resids)
    Σ = cov(resids)
    D = sqrt(Diagonal(Σ))
    iD = inv(D)
    R = iD * Σ * iD
    R = (R + R') / 2
    f = x -> LL2step(x, R, resids, p, q)
    res = optimize(x->-sum(f(x)), [.05, .9], BFGS(), autodiff=:forward)
    x = Optim.minimizer(res)
    Hpp = ForwardDiff.hessian(x->sum(f(x)), x)/n
    g = x -> sum(LL2step_full(x, R, data, p, q))
    np = 2 + dim * nparams(GARCH{1, 1})
    coefs = zeros(np)
    coefs[1:2] .= x
    Htt = zeros(np-2, np-2)
    dt = zeros(n, np-2)
    for i = 1:dim
        w=1+(i-1)*nparams(GARCH{1, 1}):1+i*nparams(GARCH{1, 1})-1
        coefs[2 .+ w] .= univariatespecs[i].spec.coefs
        Htt[w, w] .= -informationmatrix(univariatespecs[i], expected=false)
        dt[:, w] = scores(univariatespecs[i])
    end
    Hpt = ForwardDiff.hessian(g, coefs)[1:2, 3:end]/n
    h = (x, y) -> LL2step_full(x, y, R, data, p, q)
    hh = y -> ForwardDiff.gradient(x->h(x, y), coefs[1:2])
    Hpt2 = ForwardDiff.jacobian(hh, coefs[3:end])

    dp = ForwardDiff.jacobian(f, x)
    A = dp-(Hpt*inv(Htt)*dt')'
    C = inv(Hpp)*A'*A*inv(Hpp)/n^2
    @show sqrt(Diagonal(C))
    @show sqrt(Diagonal(inv(Hpp))/n)
    return DCC{p, q}(R, x, univariatespecs)
end

#LC(Θ, ϕ) in Engle (2002)
function LL2step_full(dcccoef::Array{T}, garchcoef::Array{T2}, R, data, p, q) where {T, T2}
    n, dims = size(data)
    resids = Array{T2}(undef, size(data))
    for i = 1:dims
        params = garchcoef[1+(i-1)*nparams(GARCH{1, 1}):1+i*nparams(GARCH{1, 1})-1]
        ht = T2[]
        lht = T2[]
        zt = T2[]
        at = T2[]
        loglik!(ht, lht, zt, at, GARCH{1, 1, Float64}, StdNormal{Float64}, NoIntercept(), data[:, i], params)
        resids[:, i] = zt

    end

    LL2step(dcccoef, R, resids, p, q)
end

#LC(Θ, ϕ) in Engle (2002)
function LL2step_full(coef::Array{T}, R, data, p, q) where {T}
    n, dims = size(data)
    resids = Array{T}(undef, size(data))
    for i = 1:dims
        params = coef[3+(i-1)*nparams(GARCH{1, 1}):3+i*nparams(GARCH{1, 1})-1]
#        s2 = var(data[:, i])
#        for t=1:n
#            d = data[t, i]
#            resids[t, i] = d/s2
#            s2 = params[1]+params[2]*s2+params[3]*d
#        end
        ht = T[]
        lht = T[]
        zt = T[]
        at = T[]
        loglik!(ht, lht, zt, at, GARCH{1, 1, Float64}, StdNormal{Float64}, NoIntercept(), data[:, i], params)
        resids[:, i] = zt
    end

    LL2step(coef[1:2], R, resids, p, q)
end

#LC(Θ_hat, ϕ) in Engle (2002)
function LL2step(coef::Array{T}, R, resids::Array{T2}, p, q) where {T, T2}
    all([0, 0] .< coef .< [1, 1]) || return T(-Inf)
    n, dims = size(resids)
    LL = zeros(T2, n)
    @show T
    @show T2
    a = coef[1]
    b = coef[2]
    abs(a+b)>1 && return T(-Inf)
    e = resids[1, :]
    Rt = copy(R)

    for t=1:n
        if t > max(p, q)
            Rt = R * (1-a-b) + a * Symmetric(e*e') + b * Rt
            RD5 = inv(sqrt(Diagonal(Rt)))
            Rt = RD5 * Rt * RD5
            Rt = (Rt + Rt')/2
        end
        e .= resids[t, :]
        C = cholesky(Rt, check=false).L
        u = inv(C) * e
        L = (dot(e, e) - dot(u, u))/2-logdet(C)
        if t==1 @show L, eltype(LL); end
        LL[t] = L
    end
    LL
end
