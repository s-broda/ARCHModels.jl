__precompile__()
#Todo:
#pass instances of GARCH{1,1} so we can enforce invariants?
#Docs
#CI

module ARCH

using StatsBase: StatisticalModel

import StatsBase: loglikelihood, nobs, fit
export loglikelihood, nobs, fit
export ARCHModel, VolatilitySpec, simulate
const FP = AbstractFloat
abstract type VolatilitySpec end

struct ARCHModel{VS<:VolatilitySpec, T<:AbstractFloat, df} <: StatisticalModel
  data::Vector{T}
  coefs::NTuple{df,T}
  ARCHModel{VS, T, df}(data, coefs) where {VS, T, df}=new(data, coefs)
end
ARCHModel{T1<:VolatilitySpec, T2, df}(VS::Type{T1}, data::Vector{T2}, coefs::NTuple{df,T2}) = ARCHModel{VS, T2, df}(data, coefs)

loglikelihood{T}(am::ARCHModel{T}) = arch_loglik!(T, am.data, zeros(am.data), am.coefs...)
nobs(am::ARCHModel) = length(am.data)
fit{T}(AM::Type{ARCHModel{T}}, data) = fit(T,data)

function simulate{T1<:VolatilitySpec, T2<:AbstractFloat, N}(VS::Type{T1}, nobs, coefs::NTuple{N,T2})
  const warmup = 100
  data = zeros(T2, nobs+warmup)
  ht = zeros(T2, nobs+warmup)
  archsim!(VS, data, ht, coefs...)
  data[warmup+1:warmup+nobs]
end

include("GARCH.jl")
end#module
