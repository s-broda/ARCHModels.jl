__precompile__()
#Todo:

#pass instances of GARCH{1,1} so we can enforce invariants? then if the instance knows r and p, we can simplify selectinputs and selectmode
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

"""
abstract type VolatilitySpec end
"""
abstract type VolatilitySpec end

"""
    struct ARCHModel{VS<:VolatilitySpec, T<:AbstractFloat, df} <: StatisticalModel
"""
struct ARCHModel{VS<:VolatilitySpec, T<:AbstractFloat, df} <: StatisticalModel
    data::Vector{T}
    coefs::NTuple{df,T}
    ARCHModel{VS, T, df}(data, coefs) where {VS, T, df}=new(data, coefs)
end
ARCHModel{T1<:VolatilitySpec, T2, df}(VS::Type{T1}, data::Vector{T2}, coefs::NTuple{df,T2}) = ARCHModel{VS, T2, df}(data, coefs)

loglikelihood{T}(am::ARCHModel{T}) = _loglik!(T, am.data, zeros(am.data), [am.coefs...])
nobs(am::ARCHModel) = length(am.data)
dof(am::ARCHModel{VS, T, df}) where {VS, T, df}= df
coef(am::ARCHModel)=am.coefs

fit{T}(AM::Type{ARCHModel{T}}, data) = fit(T, data)
coefnames{spec}(::ARCHModel{spec}) = coefnames(spec)

"""
Simulate an ARCH model.
"""
function simulate{T1<:VolatilitySpec, T2<:AbstractFloat, N}(VS::Type{T1}, nobs, coefs::NTuple{N,T2})
  const warmup = 100
  data = zeros(T2, nobs+warmup)
  ht = zeros(T2, nobs+warmup)
  _sim!(VS, data, ht, [coefs...])
  data[warmup+1:warmup+nobs]
end

function _loglik!{spec<:VolatilitySpec, T1<:FP}(M::Type{spec}, data::Vector{T1}, ht::Vector{T1}, coefs::Vector{T1})
    T = length(data)
    r = _checkinputs(M, coefs, T)
    log2pi = T1(1.837877066409345483560659472811235279722794947275566825634303080965531391854519)
    @inbounds begin
        h0 = _uncond(M, coefs)
        h0 > 0 || return T1(NaN)
        lh0 = log(h0)
        ht[1:r] .= h0
        LL = r*lh0+sum(data[1:r].^2)/h0
        @fastmath for t = r+1:T
            _update!(M, data, ht, coefs, t)
            LL += log(ht[t]) + data[t]^2/ht[t]
        end#for
    end#inbounds
    LL = -(T*log2pi+LL)/2
end#function


function _sim!{spec<:VolatilitySpec, T1<:FP}(M::Type{spec}, data::Vector{T1}, ht::Vector{T1}, coefs::Vector{T1})
    T =  length(data)
    r = _checkinputs(M, coefs, T)
    h0 = _uncond(M, coefs)
    h0 > 0 || error("Model is nonstationary.")
    randn!(@view data[1:r])
    data[1:r] .*= sqrt(h0)
    @inbounds begin
        for t = r+1:T
            _update!(M, data, ht, coefs, t)
            data[t] = sqrt(ht[t])*randn(T1)
        end
    end
end

function fit{spec<:VolatilitySpec}(M::Type{spec}, data, algorithm=BFGS; kwargs...)
    ht = zeros(data)
    obj = x -> -_loglik!(M, data, ht, x)
    x0 = _start(M, data)
    lower, upper = _constraints(M, data)
    res = optimize(obj, x0, lower, upper, Fminbox{algorithm}(); kwargs...)
    return ARCHModel(M, data, Tuple(res.minimizer))
end

function selectmodel{spec<:VolatilitySpec, T}(M::Type{spec}, data::Vector{T}, maxpq=3, args...; criterion=bic, kwargs...)
    nparams=length(Base.unwrap_unionall(M).parameters) #e.g., two (p and q) for GARCH{p, q}
    res=_selectmodel(spec, Val{nparams}(), Val{maxpq}(), data)
    crits = criterion.(res)
    _, ind = findmin(crits)
    return res[ind]
end

@generated function _selectmodel{nparams, maxpq}(spec, ::Val{nparams}, ::Val{maxpq}, data)
quote
    res =Array{ARCHModel, $nparams}(@ntuple($nparams, i->$maxpq))
    @nloops $nparams i res begin
        @nref($nparams, res, i) = fit(spec{@ntuple($nparams, i)...}, data)
    end
    return res
end
end
include("GARCH.jl")
end#module
