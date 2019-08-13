struct DCC{p, q, VS<:VolatilitySpec, T<:AbstractFloat, d} <: MultivariateVolatilitySpec{T, d}
    R::Matrix{T}
    coefs::Vector{T}
    univariatespecs::Vector{VS}
    function DCC{p, q, VS, T, d}(R::Array{T}, coefs::Vector{T}, univariatespecs:: Vector{VS}) where {p, q, T, VS<:VolatilitySpec, d}
        length(coefs) == nparams(DCC{p, q})  || throw(NumParamError(nparams(DCC{p, q}), length(coefs)))
        @assert d == length(univariatespecs)
        new{p, q, VS, T, d}(R, coefs, univariatespecs)
    end
end
DCC{p, q}(R::Matrix{T}, coefs::Vector{T}, univariatespecs::Vector{VS}) where {p, q, T, VS<:VolatilitySpec{T}} = DCC{p, q, VS, T, length(univariatespecs)}(R, coefs, univariatespecs)

nparams(::Type{DCC{p, q}}) where {p, q} = p+q

fit(::Type{<:DCC}, data; method=:largescale) = fit(DCC{1, 1}, data; method=method)

fit(DCCspec::Type{<:DCC{p, q}}, data; method=:largescale) where {p, q} = fit(DCC{p, q, GARCH{1, 1}}, data; method=method)

function fit(DCCspec::Type{<:DCC{p, q, VS}}, data::Matrix{T}; method=:largescale) where {p, q, VS<: VolatilitySpec, T, d}
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
    method == :largescale ? Σ = analytical_shrinkage(resids) : Σ = cov(resids)
    D = sqrt(Diagonal(Σ))
    iD = inv(D)
    R = iD * Σ * iD
    R = (R + R') / 2
    np = 2 + dim * nparams(GARCH{1, 1})
    coefs = zeros(np)
    Htt = zeros(np-2, np-2)
    dt = zeros(n, np-2)
    for i = 1:dim
        w=1+(i-1)*nparams(GARCH{1, 1}):1+i*nparams(GARCH{1, 1})-1
        coefs[2 .+ w] .= univariatespecs[i].spec.coefs
        Htt[w, w] .= -informationmatrix(univariatespecs[i], expected=false)
        dt[:, w] = scores(univariatespecs[i])
    end
    if method == :twostep
        f = x -> LL2step(x, R, resids, p, q)
        res = optimize(x->-sum(f(x)), [.05, .9], BFGS(), autodiff=:forward)
        @show x = Optim.minimizer(res)
        coefs[1:2] .= x
        Hpp = ForwardDiff.hessian(x->sum(f(x)), x)/n
        dp = ForwardDiff.jacobian(f, x)

        # g = x -> sum(LL2step_full(x, R, data, p, q))
        # Hpt = ForwardDiff.hessian(g, coefs)[1:2, 3:end]/n
        # use finite differences instead, because we don't need the whole
        # Hessian, and I couldn't figure out how to do this with ForwardDiff
        g = (x, y) -> sum(LL2step_full(x, y, R, data, p, q))
        dg = x -> ForwardDiff.gradient(y->g(x, y), coefs[3:end])/n
        h = 1e-7
        e1 = [h, 0]
        e2 = [0, h]
        dg0 = dg(x)
        j1 = (dg(x+e1)-dg0)/h
        j2 = (dg(x+e2)-dg0)/h
        Hpt = [j1'; j2']
        A = dp-(Hpt*inv(Htt)*dt')'
        C = inv(Hpp)*A'*A*inv(Hpp)/n^2
        @show std = sqrt(Diagonal(C))
    elseif method==:largescale
        f = x -> LL2step_pairs(x, R, resids, p, q)
        res = optimize(x->-sum(f(x)), [.05, .9], BFGS(), autodiff=:forward)
        @show x = Optim.minimizer(res)
        coefs[1:2] = x
        g = x -> LL2step_pairs(x, R, resids, p, q, true)
        sc = ForwardDiff.jacobian(g, x)
        I = sc'*sc/n/(dim-1)

        h = x-> LL2step_pairs_full(x, R, data, p, q)
        H = ForwardDiff.hessian(x->sum(h(x)), coefs)/n/(dim-1)
        J = H[1:2, 1:2] - H[1:2, 3:end] * inv(H[3:end, 3:end]) * H[1:2, 3:end]'
        @show std = sqrt.(diag(inv(J)*I*inv(J))/n) # from the 2014 version of the paper

        as = hcat(dt, sc) # all scores
        Sig = as'*as/n/dim
        Jnt = hcat(inv(H[1:2, 1:2])*H[1:2, 3:end]*inv(Htt), -inv(H[1:2, 1:2]))
        @show std2=sqrt.(diag(Jnt*Sig*Jnt'/n)) # from the 2018 version
    else error("No method :$method.")
    end
    return MultivariateARCHModel(DCC{p, q}(R, x, getproperty.(univariatespecs, :spec)), data)
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
    LL2step2(dcccoef, R, resids, p, q)
end

#LC(Θ, ϕ) in Engle (2002). not actually the full log-likelihood
#this method only needed for Hpt when using ForwardDiff
# function LL2step_full(coef::Array{T}, R, data, p, q) where {T}
#     n, dims = size(data)
#     resids = Array{T}(undef, size(data))
#     for i = 1:dims
#         params = coef[3+(i-1)*nparams(GARCH{1, 1}):3+i*nparams(GARCH{1, 1})-1]
#         ht = T[]
#         lht = T[]
#         zt = T[]
#         at = T[]
#         loglik!(ht, lht, zt, at, GARCH{1, 1, Float64}, StdNormal{Float64}, NoIntercept(), data[:, i], params)
#         resids[:, i] = zt
#     end
#     LL2step(coef[1:2], R, resids, p, q)
# end

#LC(Θ_hat, ϕ) in Engle (2002)
function LL2step(coef::Array{T}, R, resids::Array{T2}, p, q) where {T, T2}
    n, dims = size(resids)
    LL = zeros(T, n)
    all([0, 0] .< coef .< [1, 1]) || (fill!(LL, T(-Inf)); return LL)
    a = coef[1]
    b = coef[2]
    abs(a+b)>1 && (fill!(LL, T(-Inf)); return LL)
    e = @view resids[1, :]
    Rt = Symmetric(zeros(T, dims, dims))
    Rt .= Symmetric(R)
    RD5 = Diagonal(zeros(T, dims))
    R = Symmetric(R)
    C = cholesky(Rt).L
    u = inv(C) * e
    for t=1:n
        if t > max(p, q)
            Rt .= Symmetric(R* (1-a-b)  .+ a * Symmetric(e*e') .+ b * Rt)
            RD5 .= inv(sqrt(Diagonal(Rt)))
            Rt .= Symmetric(RD5 * Rt * RD5)
            #Rt .= (Rt + Rt')/2
            C .= cholesky(Rt).L
        end
        e = @view resids[t, :]
        u .= inv(C) * e
        L = (dot(e, e) - dot(u, u))/2-logdet(C)
        LL[t] = L
    end
    LL
end

#doall toggles whether to return all individual likelihood contributions
function LL2step_pairs(coef::Array{T}, R, resids::Array{T2}, p, q, doall=false) where {T, T2}
    n, dims = size(resids)
    len = doall ? n : 1
    LL = zeros(T, len, dims)
    Threads.@threads for k = 1:dims-1
        thell = ll(coef, R[k, k+1], resids[:, k:k+1], p, q, doall)
        if doall
            LL[:, k] .= thell
        else
            LL[1, k:k] .= thell
        end
    end
    sum(LL, dims=2)
end
function LL2step_pairs_full(coef::Array{T}, R, data, p, q) where {T, T2}
    dcccoef = coef[1:2]
    garchcoef = coef[3:end]
    n, dims = size(data)
    resids = Array{T}(undef, size(data))
    for i = 1:dims
        params = garchcoef[1+(i-1)*nparams(GARCH{1, 1}):1+i*nparams(GARCH{1, 1})-1]
        ht = T[]
        lht = T[]
        zt = T[]
        at = T[]
        loglik!(ht, lht, zt, at, GARCH{1, 1, Float64}, StdNormal{Float64}, NoIntercept(), data[:, i], params)
        resids[:, i] = zt
    end
    LL2step_pairs(dcccoef, R, resids, p, q)
end
function ll(coef::Array{T}, rho, resids, p, q, doall=false) where T
    a = coef[1]
    b = coef[2]
    0 < a < 1 || return T(-Inf)
    0 < b < 1 || return T(-Inf)
    abs(a + b) < 1 || return T(-Inf)
    n, dims = size(resids)
    f = 1 - a - b
    len = doall ? n : 1
    LL = zeros(T, len)

    rt = T(rho)
    e1 = resids[1, 1]
    e2 = resids[1, 2]
    for t=1:n
        if t > max(p, q)
            s1 = 1 + a * (e1 * e1 - 1)
            s2 = 1 + a * (e2 * e2 - 1)
            rt = rho * f + a * e1 * e2 + b * rt
            rt = rt / sqrt(s1 * s2)
        end
        e1 = resids[t, 1]
        e2 = resids[t, 2]
        r2 = rt^2
        d = 1 - r2

        L = (((e1*e1 + e2*e2) * r2 - 2 * rt *e1 * e2) / d + log(d)) / 2

        if doall
            LL[t] = -L
        else
            LL[1] -= L
        end
     end
    LL
end


# same as LL2step except for the inititalization type.
function LL2step2(coef::Array{T}, R, resids::Array{T2}, p, q) where {T, T2}
    n, dims = size(resids)
    LL = zeros(T2, n)
    all([0, 0] .< coef .< [1, 1]) || (fill!(LL, T2(-Inf)); return LL)
    a = coef[1]
    b = coef[2]
    abs(a+b)>1 && (fill!(LL, T2(-Inf)); return LL)
    e = @view resids[1, :]
    Rt = Symmetric(zeros(T2, dims, dims))
    Rt .= Symmetric(R)
    RD5 = Diagonal(zeros(T2, dims))
    R = Symmetric(R)
    C = cholesky(Rt).L
    u = inv(C) * e
    for t=1:n
        if t > max(p, q)
            Rt .= Symmetric(R* (1-a-b)  .+ a * Symmetric(e*e') .+ b * Rt)
            RD5 .= inv(sqrt(Diagonal(Rt)))
            Rt .= Symmetric(RD5 * Rt * RD5)
            #Rt .= (Rt + Rt')/2
            C .= cholesky(Rt).L
        end
        e = @view resids[t, :]
        u .= inv(C) * e
        L = (dot(e, e) - dot(u, u))/2-logdet(C)
        LL[t] = L
    end
    LL
end
