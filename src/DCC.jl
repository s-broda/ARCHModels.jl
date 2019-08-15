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

fit(::Type{<:DCC}, data::Matrix{T}; meanspec=Intercept{T}, method=:largescale) where {T} = fit(DCC{1, 1}, data; meanspec=meanspec, method=method)

fit(DCCspec::Type{<:DCC{p, q}}, data::Matrix{T}; meanspec=Intercept{T},  method=:largescale) where {p, q, T} = fit(DCC{p, q, GARCH{1, 1}}, data; meanspec=meanspec, method=method)

function fit(DCCspec::Type{<:DCC{p, q, VS}}, data::Matrix{T}; meanspec=Intercept{T}, method=:largescale) where {p, q, VS<: VolatilitySpec, T, d}
    n, dim = size(data)
    r = p + q
    resids = similar(data)
    m = fit(VS, data[:, 1], meanspec=meanspec)
    resids[:, 1] = residuals(m)
    univariatespecs = Vector{typeof(m)}(undef, dim)
    univariatespecs[1] = m
    Threads.@threads for i = 2:dim
        m = fit(VS, data[:, i], meanspec=meanspec)
        univariatespecs[i] = m
        resids[:, i] = residuals(m)
    end
    method == :largescale ? Σ = analytical_shrinkage(resids) : Σ = cov(resids)
    D = sqrt(Diagonal(Σ))
    iD = inv(D)
    R = iD * Σ * iD
    R = (R + R') / 2
    nvolaparams = nparams(VS)
    np = r + dim * nvolaparams
    coefs = zeros(np)
    Htt = zeros(np-p-q, np-p-q)
    dt = zeros(n, np-p-q)
    for i = 1:dim
        w=1+(i-1)*nvolaparams:1+i*nvolaparams-1
        coefs[r .+ w] .= univariatespecs[i].spec.coefs
        Htt[w, w] .= -informationmatrix(univariatespecs[i], expected=false)[1:nvolaparams, 1:nvolaparams]
        dt[:, w] = scores(univariatespecs[i])[:, 1:nvolaparams]
    end
    x0 = zeros(T, p+q)
    x0[1:p] .= 0.9/p
    x0[p+1:end] .= 0.05/q
    if method == :twostep
        f = x -> LL2step(DCCspec, x, R, resids)
        res = optimize(x->-sum(f(x)), x0, BFGS(), autodiff=:forward)
        @show x = Optim.minimizer(res)
        coefs[1:r] .= x
        Hpp = ForwardDiff.hessian(x->sum(f(x)), x)/n
        dp = ForwardDiff.jacobian(f, x)

        # g = x -> sum(LL2step_full(x, R, data, p, q))
        # Hpt = ForwardDiff.hessian(g, coefs)[1:2, 3:end]/n
        # use finite differences instead, because we don't need the whole
        # Hessian, and I couldn't figure out how to do this with ForwardDiff
        g = (x, y) -> sum(LL2step_full(DCCspec, VS, x, y, R, data))
        dg = x -> ForwardDiff.gradient(y->g(x, y), coefs[1+r:end])/n
        h = 1e-7
        Hpt = zeros(p+q, dim*nparams(VS))
        for j=1:p+q
            dg0 = dg(x)
            xp = copy(x); xp[j] += h
            ddg = (dg(xp)-dg0)/h
            Hpt[j, :] = ddg
        end
        A = dp-(Hpt*inv(Htt)*dt')'
        C = inv(Hpp)*A'*A*inv(Hpp)/n^2
        @show std = sqrt.(diag(C))
    elseif method==:largescale
        f = x -> LL2step_pairs(DCCspec, x, R, resids)
        res = optimize(x->-sum(f(x)), x0, BFGS(), autodiff=:forward)
        @show x = Optim.minimizer(res)
        #return
        coefs[1:r] = x
        g = x -> LL2step_pairs(DCCspec, x, R, resids, true)
        sc = ForwardDiff.jacobian(g, x)
        I = sc'*sc/n/dim

        h = x-> LL2step_pairs_full(DCCspec, VS, x, R, data)
        H = ForwardDiff.hessian(x->sum(h(x)), coefs)/n/dim
        J = H[1:r, 1:r] - H[1:r, r+1:end] * inv(H[1+r:end, 1+r:end]) * H[1:r, 1+r:end]'
        @show std = sqrt.(diag(inv(J)*I*inv(J))/n) # from the 2014 version of the paper
        as = hcat(dt, sc) # all scores
        Sig = as'*as/n/dim
        Jnt = hcat(inv(H[1:r, 1:r])*H[1:r, 1+r:end]*inv(Htt), -inv(H[1:r, 1:r]))
        @show std=sqrt.(diag(Jnt*Sig*Jnt'/n)) # from the 2018 version
    else error("No method :$method.")
    end
    return MultivariateARCHModel(DCC{p, q}(R, x, getproperty.(univariatespecs, :spec)), data, MultivariateStdNormal{T, dim}(), getproperty.(univariatespecs, :meanspec), true)
end

#LC(Θ, ϕ) in Engle (2002)
function LL2step_full(DCCspec::Type{<:DCC{p, q}}, VS, dcccoef::Array{T}, garchcoef::Array{T2}, R, data) where {T, T2, p, q}
    n, dims = size(data)
    resids = Array{T2}(undef, size(data))
    for i = 1:dims
        params = garchcoef[1+(i-1)*nparams(VS):1+i*nparams(VS)-1]
        ht = T2[]
        lht = T2[]
        zt = T2[]
        at = T2[]
        loglik!(ht, lht, zt, at, VS, StdNormal{Float64}, NoIntercept(), data[:, i], params)
        resids[:, i] = zt
    end
    LL2step2(DCCspec, dcccoef, R, resids)
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
function LL2step(DCCspec::Type{<:DCC{p, q}}, coef::Array{T}, R, resids::Array{T2}) where {T, T2, p, q}
    n, dims = size(resids)
    LL = zeros(T, n)
    all(0 .< coef .< 1) || (fill!(LL, T(-Inf)); return LL)
    abs(sum(coef))>1 && (fill!(LL, T(-Inf)); return LL)
    f = 1 - sum(coef)


    e = @view resids[1, :]
    Rt = [zeros(T, dims, dims) for _ in 1:n]
    R = Symmetric(R)
    Rt[1:max(p,q)] .= [R for _ in 1:max(p,q)]
    RD5 = Diagonal(zeros(T, dims))
    C = cholesky(Rt[1]).L
    u = inv(C) * e
    for t=1:n
        if t > max(p, q)
            Rt[t] .= R * f
            for i = 1:p
                Rt[t] .+=  coef[i] * Rt[t-i]
            end
            for i = 1:q
                Rt[t] .+= coef[p+i]  * resids[t-i, :]*resids[t-i, :]'
            end
            RD5 .= inv(sqrt(Diagonal(Rt[t])))
            Rt[t] .= Symmetric(RD5 * Rt[t] * RD5)
            C .= cholesky(Rt[t]).L
        end
        e = @view resids[t, :]
        u .= inv(C) * e
        L = (dot(e, e) - dot(u, u))/2-logdet(C)
        LL[t] = L
    end
    LL
end

#same as LL2step, except for init type
function LL2step2(DCCspec::Type{<:DCC{p, q}}, coef::Array{T2}, R, resids::Array{T}) where {T, T2, p, q}
    n, dims = size(resids)
    LL = zeros(T, n)
    all(0 .< coef .< 1) || (fill!(LL, T(-Inf)); return LL)
    abs(sum(coef))>1 && (fill!(LL, T(-Inf)); return LL)
    f = 1 - sum(coef)


    e = @view resids[1, :]
    Rt = [zeros(T, dims, dims) for _ in 1:n]
    R = Symmetric(R)
    Rt[1:max(p,q)] .= [R for _ in 1:max(p,q)]
    RD5 = Diagonal(zeros(T, dims))
    C = cholesky(Rt[1]).L
    u = inv(C) * e
    for t=1:n
        if t > max(p, q)
            Rt[t] .= R * f
            for i = 1:p
                Rt[t] .+=  coef[i] * Rt[t-i]
            end
            for i = 1:q
                Rt[t] .+= coef[p+i]  * resids[t-i, :]*resids[t-i, :]'
            end
            RD5 .= inv(sqrt(Diagonal(Rt[t])))
            Rt[t] .= Symmetric(RD5 * Rt[t] * RD5)
            C .= cholesky(Rt[t]).L
        end
        e = @view resids[t, :]
        u .= inv(C) * e
        L = (dot(e, e) - dot(u, u))/2-logdet(C)
        LL[t] = L
    end
    LL
end

#doall toggles whether to return all individual likelihood contributions
function LL2step_pairs(DCCspec::Type{<:DCC{p, q}}, coef::Array{T}, R, resids::Array{T2}, doall=false) where {T, T2, p, q}
    n, dims = size(resids)
    len = doall ? n : 1
    LL = zeros(T, len, dims)
    #Threads.@threads
    for k = 1:dims-1
        thell = ll(DCCspec, coef, R[k, k+1], resids[:, k:k+1], doall)
        if doall
            LL[:, k] .= thell
        else
            LL[1, k:k] .= thell
        end
    end
    sum(LL, dims=2)
end
function LL2step_pairs_full(DCCspec::Type{<:DCC{p, q}}, VS::Type{<:VolatilitySpec}, coef::Array{T}, R, data) where {T, T2, p, q}
    dcccoef = coef[1:p+q]
    garchcoef = coef[p+q+1:end]
    n, dims = size(data)
    resids = Array{T}(undef, size(data))
    for i = 1:dims
        params = garchcoef[1+(i-1)*nparams(VS):1+i*nparams(VS)-1]
        ht = T[]
        lht = T[]
        zt = T[]
        at = T[]
        loglik!(ht, lht, zt, at, VS, StdNormal{Float64}, NoIntercept(), data[:, i], params)
        resids[:, i] = zt
    end
    LL2step_pairs(DCCspec::Type{<:DCC{p, q}}, dcccoef, R, resids)
end
function ll(DCCspec::Type{<:DCC{p, q}}, coef::Array{T}, rho, resids, doall=false) where {T, p, q}
    all(0 .< coef .< 1) || return T(-Inf)
    abs(sum(coef)) < 1 || return T(-Inf)
    n, dims = size(resids)
    f = 1 - sum(coef)
    len = doall ? n : 1
    LL = zeros(T, len)

    rt = zeros(T, n) # should switch this to circbuff for speed
    s1 = T(1)
    s2 = T(1)
    fill!(rt, rho)
    for t=1:n
        if t > max(p, q)
            s1 = T(1)
            s2 = T(1)
            rt[t] = rho * f
            for i = 1:q
                s1 += coef[p+i] * (resids[t-i, 1]^2 - 1)
                s2 += coef[p+i] * (resids[t-i, 2]^2 - 1)
                rt[t] += coef[p+i] * resids[t-i, 1] * resids[t-i, 2]
            end
            for i = 1:p
                rt[t] += coef[i] * rt[t-i]
            end
            rt[t] = rt[t] / sqrt(s1 * s2)
        end
        e1 = resids[t, 1]
        e2 = resids[t, 2]
        r2 = rt[t]^2
        d = 1 - r2

        L = (((e1*e1 + e2*e2) * r2 - 2 * rt[t] *e1 * e2) / d + log(d)) / 2

        if doall
            LL[t] = -L
        else
            LL[1] -= L
        end
     end
    LL
end
