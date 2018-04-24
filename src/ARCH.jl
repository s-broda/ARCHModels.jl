__precompile__()
#Todo:

#change coefs to vectors instead of tuples?
#pretty print output by overloading show
#change to where syntax everywhere
#plotting via timeseries
#marketdata
#alternative error distributions
#standard errors
#demean?
# don't pass data into start
#figure out what to do about unid'd models. Eg, in fit, we had
    #without ARCH terms, volatility is constant and beta_i is not identified.
    #q == 0 && return ARCHModel(G, data, Tuple([mean(data.^2); zeros(T, p)]))

module ARCH

using StatsBase: StatisticalModel
using Optim
using Base.Cartesian: @nloops, @nref, @ntuple
export BFGS
import StatsBase: loglikelihood, nobs, fit, aic, bic, aicc, dof, coef, coefnames
export            loglikelihood, nobs, fit, aic, bic, aicc, dof, coef, coefnames
export ARCHModel, VolatilitySpec, simulate, selectmodel
const FP = AbstractFloat

abstract type VolatilitySpec end

struct ARCHModel{VS<:VolatilitySpec, T<:AbstractFloat, df} <: StatisticalModel
    data::Vector{T}
    coefs::NTuple{df,T}
    ARCHModel{VS, T, df}(data, coefs) where {VS, T, df}=new(data, coefs)
end
ARCHModel(VS::Type{T1}, data::Vector{T2}, coefs::NTuple{df,T2}) where {T1<:VolatilitySpec, T2, df} = ARCHModel{VS, T2, df}(data, coefs)

loglikelihood(am::ARCHModel{T}) where {T} = loglik!(T, am.data, zeros(am.data), [am.coefs...])
nobs(am::ARCHModel) = length(am.data)
dof(am::ARCHModel{VS, T, df}) where {VS, T, df}= df
coef(am::ARCHModel)=am.coefs

fit(AM::Type{ARCHModel{T}}, data) where {T} = fit(T, data)
coefnames(::ARCHModel{spec}) where {spec} = coefnames(spec)

function simulate(VS::Type{T1}, nobs, coefs::NTuple{N,T2}) where {T1<:VolatilitySpec, T2<:AbstractFloat, N}
  const warmup = 100
  data = zeros(T2, nobs+warmup)
  ht = zeros(T2, nobs+warmup)
  sim!(VS, data, ht, [coefs...])
  data[warmup+1:warmup+nobs]
end

function loglik!(M::Type{spec}, data::Vector{T1}, ht::Vector{T1}, coefs::Vector{T1}) where {spec<:VolatilitySpec, T1<:FP}
    T = length(data)
    r = presample(M)
    length(coefs) == nparams(M) || error("Incorrect number of parameters: expected $(p+q+1), got $(length(coefs)).")
    T > r || error("Sample too small.")
    log2pi = T1(1.837877066409345483560659472811235279722794947275566825634303080965531391854519)
    @inbounds begin
        h0 = uncond(M, coefs)
        h0 > 0 || return T1(NaN)
        lh0 = log(h0)
        ht[1:r] .= h0
        LL = r*lh0+sum(data[1:r].^2)/h0
        @fastmath for t = r+1:T
            update!(M, data, ht, coefs, t)
            LL += log(ht[t]) + data[t]^2/ht[t]
        end#for
    end#inbounds
    LL = -(T*log2pi+LL)/2
end#function


function sim!(M::Type{spec}, data::Vector{T1}, ht::Vector{T1}, coefs::Vector{T1}) where {spec<:VolatilitySpec, T1<:FP}
    T =  length(data)
    r = presample(M)
    length(coefs) == nparams(M) || error("Incorrect number of parameters: expected $(p+q+1), got $(length(coefs)).")
    h0 = uncond(M, coefs)
    h0 > 0 || error("Model is nonstationary.")
    randn!(@view data[1:r])
    data[1:r] .*= sqrt(h0)
    @inbounds begin
        for t = r+1:T
            update!(M, data, ht, coefs, t)
            data[t] = sqrt(ht[t])*randn(T1)
        end
    end
end

function fit(M::Type{spec}, data, algorithm=BFGS; kwargs...) where {spec<:VolatilitySpec}
    ht = zeros(data)
    obj = x -> -loglik!(M, data, ht, x)
    x0 = startingvals(M, data)
    lower, upper = constraints(M, data)
    res = optimize(obj, x0, lower, upper, Fminbox{algorithm}(); kwargs...)
    return ARCHModel(M, data, Tuple(res.minimizer))
end

function selectmodel(M::Type{spec}, data::Vector{T}, maxpq=3, args...; criterion=bic, kwargs...) where {spec<:VolatilitySpec, T}
    ndims=length(Base.unwrap_unionall(M).parameters) #e.g., two (p and q) for GARCH{p, q}
    res=_selectmodel(spec, Val{ndims}(), Val{maxpq}(), data)
    crits = criterion.(res)
    _, ind = findmin(crits)
    return res[ind]
end

@generated function _selectmodel(spec, ::Val{ndims}, ::Val{maxpq}, data) where {ndims, maxpq}
quote
    res =Array{ARCHModel, $ndims}(@ntuple($ndims, i->$maxpq))
    @nloops $ndims i res begin
        @nref($ndims, res, i) = fit(spec{@ntuple($ndims, i)...}, data)
    end
    return res
end
end
include("GARCH.jl")
end#module
