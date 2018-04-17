__precompile__()
#Todo:
#share code between sim and loglik
#pass instances of GARCH{1,1} so we can enforce invariants?
#change coefs to vectors instead of tuples?
#pretty print output by overloading show
#change to where syntax everywhere
#plotting via timeseries
#marketdata
#alternative error distributions
#standard errors
#demean?

module ARCH

using StatsBase: StatisticalModel
using Optim
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

loglikelihood{T}(am::ARCHModel{T}) = arch_loglik!(T, am.data, zeros(am.data), [am.coefs...])
nobs(am::ARCHModel) = length(am.data)
dof(am::ARCHModel{VS, T, df}) where {VS, T, df}= df
coef(am::ARCHModel)=am.coefs

fit{T}(AM::Type{ARCHModel{T}}, data) = fit(T, data)
"""
Simulate an ARCH model.
"""
function simulate{T1<:VolatilitySpec, T2<:AbstractFloat, N}(VS::Type{T1}, nobs, coefs::NTuple{N,T2})
  const warmup = 100
  data = zeros(T2, nobs+warmup)
  ht = zeros(T2, nobs+warmup)
  archsim!(VS, data, ht, [coefs...])
  data[warmup+1:warmup+nobs]
end

include("GARCH.jl")
end#module
