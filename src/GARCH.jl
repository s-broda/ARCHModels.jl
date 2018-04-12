export GARCH
struct GARCH{p,q} <: VolatilitySpec end

function arch_loglik!{p, q, T1<:FP}(M::Type{GARCH{p,q}}, data::Vector{T1}, ht::Vector{T1}, coefs::Vector{T1})
    r = max(p, q)
    length(coefs) == p+q+1 || error("Incorrect number of parameters: expected $(p+q+1), got $(length(coefs)).")
    T = length(data)
    T > r || error("Sample too small.")
    log2pi = T1(1.837877066409345483560659472811235279722794947275566825634303080965531391854519)
    @inbounds begin
        den=one(coefs[1])
        for i = 1:p+q
            den -= coefs[i+1]
        end
        h0 = coefs[1]/den
        h0 < 0 && return T1(NaN)
        lh0 = log(h0)
        ht[1:r] .= h0
        LL = r*lh0+sum(data[1:r].^2)/h0
        @fastmath for t = r+1:T
            ht[t] = coefs[1]
            for i = 1:p
                ht[t] += coefs[i+1]*ht[t-i]
            end
            for i = 1:q
                ht[t] += coefs[i+1+p]*data[t-i]^2
            end
            LL += log(ht[t]) + data[t]^2/ht[t]
        end#for
    end#inbounds
    LL = -(T*log2pi+LL)/2
end#function


function archsim!{p,q}(::Type{GARCH{p, q}}, data, ht, coefs)
    r = max(p,q)
    length(coefs) == p+q+1 || error("Incorrect number of parameters: expected $(p+q+1), got $(length(coefs)).")
    T=length(data)
    T > r || error("Sample too small.")
    h0 = coefs[1]/(1-sum(coefs[2:end]))
    h0 > 0 || error("Model is nonstationary.")
    randn!(@view data[1:r])
    data[1:r] .*= sqrt(h0)
    @inbounds begin
        for t = r+1:T
            ht[t] = coefs[1]
            for i = 1:p
                ht[t] += coefs[i+1]*ht[t-i]
            end
            for i = 1:q
                ht[t] += coefs[i+1+p]*data[t-i]^2
            end
            data[t] = sqrt(ht[t])*randn()
        end#for
    end#inbounds
end#function

function archstart{p, q, T}(G::Type{GARCH{p,q}}, data::Array{T})
    x0 = zeros(T, p+q+1)
    x0[2:p+1] = 0.9/p
    x0[p+2:end] = 0.05/q
    x0[1] = var(data)*(one(T)-sum(x0))
    return x0
end

function fit{p, q, T}(G::Type{GARCH{p,q}}, data::Array{T}, args...; kwargs...)
    #without ARCH terms, volatility is constant and beta_i is not identified.
    q == 0 && return ARCHModel(G, data, Tuple([mean(data.^2); zeros(T, p)]))
    ht = zeros(data)
    obj = x -> -arch_loglik!(G, data, ht, x)
    x0 = archstart(G, data)
    res = optimize(obj, x0, args...; kwargs...)
    return ARCHModel(G, data, Tuple(res.minimizer))
end

function selectmodel(G::Type{GARCH}, data, maxp=3, maxq=3, args...; criterion=bic, kwargs...)
    res = Array{ARCHModel, 2}(maxp+1, maxq+1)
    for p = 0:maxp, q = 0:maxq
        res[p+1, q+1] = fit(GARCH{p, q}, data, args...; kwargs...)
    end
    crits = criterion.(res)
    println(crits)
    _, ind = findmin(crits)
    return res[ind]
end

function coefnames{p, q}(G::ARCHModel{GARCH{p,q}})
    names = Array{String, 1}(p+q+1)
    names[1]="omega"
    names[2:p+1].=(i->"beta_$i").([1:p...])
    names[p+2:p+q+1].=(i->"alpha_$i").([1+q...])
    return names
end
