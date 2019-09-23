struct DCC{p, q, VS<:UnivariateVolatilitySpec, T<:AbstractFloat, d} <: MultivariateVolatilitySpec{T, d}
    R::Matrix{T}
    coefs::Vector{T}
    univariatespecs::Vector{VS}
    method::Symbol
    function DCC{p, q, VS, T, d}(R::Array{T}, coefs::Vector{T}, univariatespecs:: Vector{VS}, method::Symbol) where {p, q, T, VS<:UnivariateVolatilitySpec, d}
        length(coefs) == nparams(DCC{p, q})  || throw(NumParamError(nparams(DCC{p, q}), length(coefs)))
        @assert d == length(univariatespecs)
        @assert method==:twostep || method==:largescale
        new{p, q, VS, T, d}(R, coefs, univariatespecs, method)
    end
end

const CCC = DCC{0, 0}

DCC{p, q}(R::Matrix{T}, coefs::Vector{T}, univariatespecs::Vector{VS}; method::Symbol=:largescale) where {p, q, T, VS<:UnivariateVolatilitySpec{T}} = DCC{p, q, VS, T, length(univariatespecs)}(R, coefs, univariatespecs, method)

nparams(::Type{DCC{p, q}}) where {p, q} = p+q

# strange dispatch behavior. to me these methods look the same, but they aren't.

# this matches ARCHModels.presample(DCC{1,1,TGARCH{0,1,1,Float64}})
presample(::Type{DCC{p, q, VS}}) where {p, q, VS} = max(p, q, presample(VS))

# this matches ARCHModels.presample(DCC{1,1,TGARCH{0,1,1,Float64},Float64,2})
presample(::Type{DCC{p, q, VS, T, d}}) where {p, q, VS, T, d} = max(p, q, presample(VS))


fit(::Type{<:DCC}, data::Matrix{T}; meanspec=Intercept{T}, method=:largescale, algorithm=BFGS(), autodiff=:forward, kwargs...) where {T} = fit(DCC{1, 1}, data; meanspec=meanspec, method=method, algorithm=algorithm, autodiff=autodiff, kwargs...)

fit(DCCspec::Type{<:DCC{p, q}}, data::Matrix{T}; meanspec=Intercept{T},  method=:largescale, algorithm=BFGS(), autodiff=:forward, kwargs...) where {p, q, T} = fit(DCC{p, q, GARCH{1, 1}}, data; meanspec=meanspec, method=method, algorithm=algorithm, autodiff=autodiff, kwargs...)

"""
    fit(DCCspec::Type{<:DCC{p, q, VS<:UnivariateVolatilitySpec}}, data::Matrix;
        method=:largescale,  dist=MultivariateStdNormal, meanspec=Intercept,
        algorithm=BFGS(), autodiff=:forward, kwargs...)

Fit the DCC model specified by `DCCspec` to `data`. If `p` and `q` or `VS` are
unspecified, then these default to 1, 1, and `GARCH{1, 1}`.

# Keyword arguments:
- `method`: one of `:largescale` or `twostep`
- `dist`: the error distribution.
- `meanspec`: the mean specification, as a type.
- `algorithm, autodiff, kwargs, ...`: passed on to the optimizer.

# Example: DCC{1, 1, GARCH{1, 1}} model:
```jldoctest
julia> fit(DCC, DOW29)

29-dimensional DCC{1, 1} - TGARCH{0,1,1} - Intercept{Float64} specification, T=2785.

DCC parameters, estimated by largescale procedure:
────────────────────
       β₁         α₁
────────────────────
  0.88762  0.0568001
────────────────────

Calculating standard errors is expensive. To show them, use
`show(IOContext(stdout, :se=>true), <model>)`
```
"""
function fit(DCCspec::Type{<:DCC{p, q, VS}}, data::Matrix{T}; meanspec=Intercept{T}, method=:largescale, algorithm=BFGS(), autodiff=:forward, dist::Type{<:MultivariateStandardizedDistribution}=MultivariateStdNormal{T}) where {p, q, VS<: UnivariateVolatilitySpec, T, d}
    n, dim = size(data)
    resids = similar(data)
    if n<12 && method == :largescale
        error("largescale method requires n>11.")
    end

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
    R = to_corr(Σ)
    x0 = zeros(T, p+q)
    if p+q>0
        x0[1:p] .= 0.9/p
        x0[p+1:end] .= 0.05/q
        if method == :twostep
            obj = LL2step
        elseif method==:largescale
            obj = LL2step_pairs
        else
            error("No method :$method.")
        end
        f = x -> obj(DCCspec, x, R, resids)
        x = optimize(x->-sum(f(x)), x0, algorithm, autodiff=autodiff).minimizer
    else # CCC
        x = x0
    end
    return MultivariateARCHModel(DCC{p, q}(R, x, getproperty.(univariatespecs, :spec); method=method), data; dist=MultivariateStdNormal{T, dim}(), meanspec=getproperty.(univariatespecs, :meanspec), fitted=true)
end


#LC(Θ_hat, ϕ) in Engle (2002)
@inline function LL2step!(Rt::Array{Array{T, 2}, 1}, DCCspec::Type{<:DCC{p, q}}, coef::Array{T}, R, resids::Array{T2}) where {T, T2, p, q}
    n, dims = size(resids)
    LL = zeros(T, n)
    all(0 .< coef .< 1) || (fill!(LL, T(-Inf)); return LL)
    abs(sum(coef))>1 && (fill!(LL, T(-Inf)); return LL)
    f = 1 - sum(coef)
    e = @view resids[1, :]
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

function LL2step(DCCspec::Type{<:DCC{p, q}}, coef::Array{T}, R, resids::Array{T2}) where {T, T2, p, q}
    n, dims = size(resids)
    Rt = [zeros(T, dims, dims) for _ in 1:n]
    LL2step!(Rt, DCCspec, coef, R, resids)
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
    for t = 1:n
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

@inline function ll(DCCspec::Type{<:DCC{p, q}}, coef::Array{T}, rho, resids, doall=false) where {T, p, q}
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
    @inbounds for t=1:n
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


function stderror(am::MultivariateARCHModel{T, d, MVS}) where {T, d, p, q, VS, MVS<:DCC{p, q, VS}}
    n, dim = size(am.data)
    r = p + q
    resids = similar(am.data)
    nunivariateparams = nparams(VS) + nparams(typeof(am.meanspec[1]))
    np = r + dim * nunivariateparams
    coefs = coef(am)
    Htt = zeros(np - r, np - r)
    dt = zeros(n, np - r)
    stderrors = zeros(np)
    Threads.@threads for i = 1:dim
        m = UnivariateARCHModel(am.spec.univariatespecs[i], am.data[:, i]; meanspec=am.meanspec[i], fitted=true)
        resids[:, i] = residuals(m)
        w=1+(i-1)*nunivariateparams:1+i*nunivariateparams-1
        Htt[w, w] .= -informationmatrix(m, expected=false)
        dt[:, w] = scores(m)
        stderrors[r .+ w] = stderror(m)
    end
    if p + q > 0
        if am.spec.method == :twostep
            f = x -> LL2step(MVS, x, am.spec.R, resids)
            Hpp = ForwardDiff.hessian(x->sum(f(x)), coefs[1:r])/n
            dp = ForwardDiff.jacobian(f, coefs[1:r])

            # g = x -> sum(LL2step_full(x, R, data, p, q))
            # Hpt = ForwardDiff.hessian(g, coefs)[1:2, 3:end]/n
            # use finite differences instead, because we don't need the whole
            # Hessian, and I couldn't figure out how to do this with ForwardDiff
            g = (x, y) -> sum(LL2step_full(MVS, VS, am.meanspec, x, y, am.spec.R, am.data))
            dg = x -> ForwardDiff.gradient(y->g(x, y), coefs[1+r:end])/n
            h = 1e-7
            Hpt = zeros(p+q, dim * nunivariateparams)
            for j=1:p+q
                dg0 = dg(coefs[1:r])
                xp = copy(coefs[1:r]); xp[j] += h
                ddg = (dg(xp)-dg0)/h
                Hpt[j, :] = ddg
            end
            A = dp-(Hpt*inv(Htt)*dt')'
            C = inv(Hpp)*A'*A*inv(Hpp)/n^2
            stderrors[1:r] = sqrt.(diag(C))
        elseif am.spec.method==:largescale
            g = x -> LL2step_pairs(MVS, x, am.spec.R, resids, true)
            sc = ForwardDiff.jacobian(g, coefs[1:r])
            I = sc'*sc/n/dim
            h = x-> LL2step_pairs_full(MVS, VS, am.meanspec, x, am.spec.R, am.data)
            H = ForwardDiff.hessian(x->sum(h(x)), coefs)/n/dim
            #J = H[1:r, 1:r] - H[1:r, r+1:end] * inv(H[1+r:end, 1+r:end]) * H[1:r, 1+r:end]'
            #std = sqrt.(diag(inv(J)*I*inv(J))/n) # from the 2014 version of the paper
            as = hcat(dt, sc) # all scores
            Sig = as'*as/n/dim
            Jnt = hcat(inv(H[1:r, 1:r])*H[1:r, 1+r:end]*inv(Htt), -inv(H[1:r, 1:r]))
            stderrors[1:r] .= sqrt.(diag(Jnt*Sig*Jnt'/n)) # from the 2018 version
        end
    end
    return stderrors
end



#LC(Θ, ϕ) in Engle (2002)
function LL2step_full(DCCspec::Type{<:DCC{p, q}}, VS, meanspec, dcccoef::Array{T}, garchcoef::Array{T2}, R, data) where {T, T2, p, q}
    n, dims = size(data)
    resids = Array{T2}(undef, size(data))
    nunivariateparams = nparams(VS) + nparams(typeof(meanspec[1]))
    for i = 1:dims
        params = garchcoef[1+(i-1)*nunivariateparams:1+i*nunivariateparams-1]
        ht = T2[]
        lht = T2[]
        zt = T2[]
        at = T2[]
        loglik!(ht, lht, zt, at, VS, StdNormal{Float64}, meanspec[i], data[:, i], params)
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

function LL2step_pairs_full(DCCspec::Type{<:DCC{p, q}}, VS::Type{<:UnivariateVolatilitySpec}, meanspec, coef::Array{T}, R, data) where {T, T2, p, q}
    dcccoef = coef[1:p+q]
    garchcoef = coef[p+q+1:end]
    n, dims = size(data)
    resids = Array{T}(undef, size(data))
    nunivariateparams = nparams(VS) + nparams(typeof(meanspec[1]))
    for i = 1:dims
        params = garchcoef[1+(i-1)*nunivariateparams:1+i*nunivariateparams-1]
        ht = T[]
        lht = T[]
        zt = T[]
        at = T[]
        loglik!(ht, lht, zt, at, VS, StdNormal{Float64}, meanspec[i], data[:, i], params)
        resids[:, i] = zt
    end
    LL2step_pairs(DCCspec::Type{<:DCC{p, q}}, dcccoef, R, resids)
end


function coefnames(::Type{<:DCC{p, q}}) where {p, q}
    names = Array{String, 1}(undef, p + q)
    names[1:p] .= (i -> "β"*subscript(i)).([1:p...])
    names[p+1:p+q] .= (i -> "α"*subscript(i)).([1:q...])
    return names
end

function coef(spec::DCC{p, q, VS, T, d})  where {p, q, VS, T, d}
    vcat(spec.coefs, [spec.univariatespecs[i].coefs for i in 1:d]...)
end

function coef(am::MultivariateARCHModel{T, d, MVS}) where {T, d, MVS<:DCC}
    vcat(am.spec.coefs, [vcat(am.spec.univariatespecs[i].coefs, am.meanspec[i].coefs) for i in 1:d]...)
end

function coefnames(am::MultivariateARCHModel{T, d, MVS}) where {T, d, p, q, VS, MVS<:DCC{p, q, VS}}
    nunivariateparams = nparams(VS) + nparams(typeof(am.meanspec[1]))
    names = Array{String, 1}(undef, p + q + d * nunivariateparams)
    names[1:p+q] .= coefnames(MVS)
    for i = 1:d
            names[p + q + 1 + (i-1) * nunivariateparams : p + q +  i * nunivariateparams] = vcat(coefnames(VS) .* subscript(i), coefnames(am.meanspec[i]) .* subscript(i))
    end
    return names
end

modname(::Type{DCC{p, q, VS, T, d}})  where {p, q, VS, T, d} = "DCC{$p, $q, $(modname(VS))}"

function show(io::IO, am::MultivariateARCHModel{T, d, MVS}) where {T, d, p, q, VS, MVS<:DCC{p, q, VS}}
    r = p + q
    cc = coef(am)[1:r]
    println(io, "\n", "$d-dimensional DCC{$p, $q} - $(modname(VS)) - $(modname(typeof(am.meanspec[1]))) specification, T=", nobs(am), ".\n")
    if isfitted(am) && (:se=>true) in io
        se = stderror(am)[1:r]
        z = cc ./ se
        if p + q >0
            println(io, "DCC parameters, estimated by $(am.spec.method) procedure:", "\n",
    	            CoefTable(hcat(cc, se, z, 2.0 * normccdf.(abs.(z))),
    	                      ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
    	                      coefnames(MVS), 4
    	                      )
    	            )
        end
    else
        if p + q > 0
            println(io, "DCC parameters", isfitted(am) ? ", estimated by $(am.spec.method) procedure:" : "", "\n",
    	            CoefTable(cc, coefnames(MVS), [""])
    	            )
            if isfitted(am)
                println(io, "\n","""Calculating standard errors is expensive. To show them, use
                 `show(IOContext(stdout, :se=>true), <model>)`""")
            end
        end
    end
end

"""
    correlations(am::MultivariateARCHModel)
Return the `nobs(am)`` estimated conditional correlation matrices.
"""
function correlations(am::MultivariateARCHModel{T, d, MVS}) where {T, d, MVS<:DCC}
    resids = residuals(am; decorrelated=false)
    n, dims = size(resids)
    Rt = [zeros(T, dims, dims) for _ in 1:n]
    LL2step!(Rt, MVS, am.spec.coefs, am.spec.R, resids)
    return Rt
end

"""
    covariances(am::MultivariateARCHModel)
Return the `nobs(am)`` estimated conditional covariance matrices.
"""
function covariances(am::MultivariateARCHModel{T, d, MVS}) where {T, d, MVS<:DCC}
    n, dims = size(am.data)
    Rt = correlations(am)
    for i = 1:d
        v = volatilities(UnivariateARCHModel(am.spec.univariatespecs[i], am.data[:, i]; meanspec=am.meanspec[i], fitted=true))
        @inbounds for t = 1:n # this is ugly, but I couldn't figure out how to do this w/ broadcasting
            Rt[t][i, :] *= v[t]
            Rt[t][:, i] *= v[t]
        end
    end
    return Rt
end

"""
    residuals(am::MultivariateARCHModel; standardized = true, decorrelated = true)
Return the residuals.
"""
function residuals(am::MultivariateARCHModel{T, d, MVS}; standardized = true, decorrelated = true) where {T, d, MVS<:DCC}
    n, dims = size(am.data)
    resids = similar(am.data)
    Threads.@threads for i = 1:dims
        m = UnivariateARCHModel(am.spec.univariatespecs[i], am.data[:, i]; meanspec=am.meanspec[i], fitted=true)
        resids[:, i] = residuals(m; standardized=standardized)
    end
    if decorrelated
        Rt = standardized ? correlations(am) : covariances(am)
        @inbounds for t = 1:n
            resids[t, :] = inv(cholesky(Rt[t]; check=false).L) * resids[t, :]
        end
    end
    return resids
end

#this assumes Ht, Rt, zt, and at are circularbuffers or vectors of arrays
Base.@propagate_inbounds @inline function update!(Ht, Rt, H, R, zt, at, MVS::Type{DCC{p, q, VS, T, d}}, coefs) where {p, q, VS, T, d}
    nvolaparams = nparams(VS)
    h5s = zeros(T, d)
    for i = 1:d
        ht = getindex.(Ht, i, i)
        lht = log.(ht)
        update!(ht, lht, getindex.(zt, i), getindex.(at, i), VS, coefs[p + q + 1 + (i-1) * nvolaparams : p + q + i * nvolaparams])
        h5s[i] = sqrt(ht[end])
    end
    Rtemp = R * (1-sum(coefs[1:p+q]))
    for i = 1:p
        Rtemp .+=  coefs[i] * Rt[end-i+1]
    end
    for i = 1:q
        Rtemp .+= coefs[p+i]  * zt[end-i+1] * zt[end-i+1]'
    end
    push!(Rt, to_corr(Rtemp))
    H5 = diagm(0 => h5s)
    push!(Ht,  H5 * Rt[end] * H5)
end

function uncond(spec::DCC{p, q, VS, T, d}) where {p, q, VS, T, d}
    h = uncond.(typeof.(spec.univariatespecs), getproperty.(spec.univariatespecs, :coefs))
    D = diagm(0 => sqrt.(h))
    return D * spec.R * D
end
