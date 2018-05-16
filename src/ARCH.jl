__precompile__()
#Todo:
#docs
#plotting via timeseries
#marketdata
#alternative error distributions
#get rid of dependency on Distributions?
#PkgBenchmark
#demean?
#GARCH instances should carry params like distributions do (eg for simulate, but not for loglike, b/c of ForwardDiff), but then it needs to be parameterizedon float type
#
#how to export arch?
#what should simulate return?


module ARCH

using StatsBase: StatisticalModel
using StatsFuns: normccdf, normlogpdf, log2π, RFunctions.tdistrand
using Optim
using ForwardDiff
using Distributions
using Roots

import Distributions: logpdf
import Base: show, showerror, Random.rand, eltype
import StatsBase: loglikelihood, nobs, fit, fit!, adjr2, aic, bic, aicc, dof, coef, coefnames, coeftable, CoefTable, stderr
export            loglikelihood, nobs, fit, fit!, adjr2, aic, bic, aicc, dof, coef, coefnames, coeftable, CoefTable, stderr
export ARCHModel, VolatilitySpec, simulate, selectmodel, StdNormal, StdTDist

abstract type VolatilitySpec end
abstract type StandardizedDistribution <: Distribution{Univariate, Continuous} end

struct NumParamError <: Exception
    expected::Int
    got::Int
end

struct LengthMismatchError <: Exception
    length1::Int
    length2::Int
end

showerror(io::IO, e::NumParamError) = print(io, "incorrect number of parameters: expected $(e.expected), got $(e.got).")
showerror(io::IO, e::LengthMismatchError) = print(io, "length of arrays does not match: $(e.length1) and $(e.length2).")

struct ARCHModel{VS<:VolatilitySpec, T<:AbstractFloat} <: StatisticalModel
    data::Vector{T}
    ht::Vector{T}
    coefs::Vector{T}
    function ARCHModel{VS, T}(data, ht, coefs) where {VS, T}
        length(coefs) == nparams(VS)  || throw(NumParamError(nparams(VS), length(coefs)))
        length(data) == length(ht)  || throw(LengthMismatchError(length(data), length(ht)))
        new(copy(data), copy(ht), copy(coefs))
    end
end
ARCHModel(::Type{VS}, data::Vector{T}, ht::Vector{T}, coefs::Vector{T}) where {VS<:VolatilitySpec, T} = ARCHModel{VS, T}(data, ht, coefs)
ARCHModel(::Type{VS}, data, coefs) where {VS<:VolatilitySpec} = (ht = zeros(data); loglik!(ht, VS, data, coefs); ARCHModel(VS, data, ht, coefs))

loglikelihood(am::ARCHModel{VS}) where {VS<:VolatilitySpec} = loglik!(zeros(am.data), VS, am.data, am.coefs)
nobs(am::ARCHModel) = length(am.data)
dof(am::ARCHModel{VS}) where {VS<:VolatilitySpec} = nparams(VS)
coef(am::ARCHModel)=am.coefs
coefnames(::ARCHModel{VS}) where {VS<:VolatilitySpec} = coefnames(VS)

function simulate(::Type{VS}, nobs, coefs::Vector{T}, dist::StandardizedDistribution=StdNormal{T}()) where {VS<:VolatilitySpec, T<:AbstractFloat}
    const warmup = 100
    data = zeros(T, nobs+warmup)
    ht = zeros(T, nobs+warmup)
    sim!(ht, VS, dist, data, coefs)
    data[warmup+1:warmup+nobs]
end
function splitcoefs(coefs, VS, SD)
    ng = nparams(VS)
    nd = nparams(SD)
    length(coefs) == ng+nd || throw(NumParamError(nparams(VS), length(coefs)))
    garchcoefs = coefs[1:ng]
    distcoefs = coefs[ng+1:ng+nd]
    return garchcoefs, distcoefs
end
function loglik!(ht::Vector{T2}, ::Type{VS}, ::Type{SD}, data::Vector{<:AbstractFloat}, coefs::Vector{T2}) where {VS<:VolatilitySpec, SD<:StandardizedDistribution, T2}
    T = length(data)
    r = presample(VS)
    garchcoefs, distcoefs = splitcoefs(coefs, VS, SD)
    T > r || error("Sample too small.")
    @inbounds begin
        h0 = uncond(VS, garchcoefs)
        h0 > 0 || return T2(NaN)
        lh0 = log(h0)
        ht[1:r] .= h0
        LL = zero(T2)
        @fastmath for t = 1:T
            t > r && update!(ht, VS, data, garchcoefs, t)
            LL += -log(ht[t])/2 + logkernel(SD, data[t]/sqrt(ht[t]), distcoefs)
        end#for
    end#inbounds
    LL += T*logconst(SD, distcoefs)
end#function

function logliks(spec, dist, data, coefs::Vector{T}) where {T}
    garchcoefs, distcoefs = splitcoefs(coefs, spec, dist)
    ht = zeros(T, length(data))
    loglik!(ht, spec, dist, data, coefs)
    LLs = -log.(ht)/2+logkernel.(dist, data./sqrt.(ht), Ref{Vector{T}}(distcoefs))+logconst(dist, distcoefs)
end

function stderr(am::ARCHModel{VS}) where {VS<:VolatilitySpec}
    f = x -> ARCH.logliks(VS, am.data, x)
    g = x -> sum(ARCH.logliks(VS, am.data, x))
    J = ForwardDiff.jacobian(f, am.coefs)
    V = J'J #outer product of scores
    Ji = -inv(ForwardDiff.hessian(g, am.coefs)) #inverse of observed Fisher information
    return sqrt.(diag(Ji*V*Ji)) #Huber sandwich
end

function sim!(ht::Vector{T1}, ::Type{VS}, dist::StandardizedDistribution, data::Vector{T1}, coefs::Vector{T1}) where {VS<:VolatilitySpec, T1<:AbstractFloat}
    T =  length(data)
    r = presample(VS)
    length(coefs) == nparams(VS) || throw(NumParamError(nparams(VS), length(coefs)))
    @inbounds begin
        h0 = uncond(VS, coefs)
        h0 > 0 || error("Model is nonstationary.")
        rand!(dist, @view data[1:r])
        data[1:r] .*= sqrt(h0)
        @fastmath for t = r+1:T
            update!(ht, VS, data, coefs, t)
            data[t] = sqrt(ht[t])*rand(dist)
        end
    end
    return nothing
end

function fit!(ht::Vector{T}, garchcoefs::Vector{T}, distcoefs::Vector{T}, ::Type{VS}, ::Type{SD}, data::Vector{T}, algorithm=BFGS; kwargs...) where {VS<:VolatilitySpec, SD<:StandardizedDistribution, T<:AbstractFloat}
    obj = x -> -loglik!(ht, VS, SD, data, x)
    lowergarch, uppergarch = constraints(VS, T)
    lowerdist, upperdist = constraints(SD, T)
    lower = vcat(lowergarch, lowerdist)
    upper = vcat(uppergarch, upperdist)
    coefs = vcat(garchcoefs, distcoefs)
    res = optimize(obj, coefs, lower, upper, Fminbox{algorithm}(); kwargs...)
    coefs .= res.minimizer
    ng = nparams(VS)
    ns = nparams(SD)
    garchcoefs .= coefs[1:ng]
    distcoefs .= coefs[ng+1:ng+ns]
    return nothing
end
fit(::Type{VS}, data::Vector{T}, ::Type{SD}=StdNormal,  algorithm=BFGS; kwargs...) where {VS<:VolatilitySpec, SD<:StandardizedDistribution, T<:AbstractFloat} = (ht = zeros(data); coefs=startingvals(VS, data); distcoefs=startingvals(SD, data); fit!(ht, coefs, distcoefs, VS, SD, data, algorithm; kwargs...); return ARCHModel(VS, SD,  data, ht, coefs))
fit!(AM::ARCHModel{VS}, algorithm=BFGS; kwargs...) where {VS<:VolatilitySpec} = (AM.coefs.=startingvals(VS, AM.data); fit!(AM.ht, AM.coefs, VS, AM.data, algorithm; kwargs...))
fit(AM::ARCHModel{VS}, algorithm=BFGS; kwargs...) where {VS<:VolatilitySpec} = (AM2=ARCHModel(VS, AM.data, AM.coefs); fit!(AM2, algorithm=BFGS; kwargs...); return AM2)

function selectmodel(::Type{VS}, data::Vector{<:AbstractFloat}, maxpq=3, args...; criterion=bic, kwargs...) where {VS<:VolatilitySpec}
    ndims =my_unwrap_unionall(VS)#e.g., two (p and q) for GARCH{p, q}
    res =Array{ARCHModel, ndims}(ntuple(i->maxpq+1, ndims))
    Threads.@threads for ind in collect(CartesianRange(size(res)))
        res[ind] = fit(VS{(ind.I .- 1)...}, data)
    end
    crits = criterion.(res)
    _, ind = findmin(crits)
    return res[ind]
end

#count the number of type vars. there's probably a better way.
function my_unwrap_unionall(a::ANY)
    count=0
    while isa(a, UnionAll)
        a = a.body
        count += 1
    end
    return count
end

function coeftable(am::ARCHModel)
    cc = coef(am)
    se = stderr(am)
    zz = cc ./ se
    CoefTable(hcat(cc, se, zz, 2.0 * normccdf.(abs.(zz))),
              ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
              coefnames(am), 4)
end

function show(io::IO, am::ARCHModel{VS}) where {VS <: VolatilitySpec}
    println(io, "\n", split("$VS", ".")[2], " model fitted to ",
        nobs(am), " observations.\n\n", coeftable(am))
end

#from here https://stackoverflow.com/questions/46671965/printing-variable-subscripts-in-julia
subscript(i::Integer) = i<0 ? error("$i is negative") : join('₀'+d for d in reverse(digits(i)))

include("standardizeddistributions.jl")
include("GARCH.jl")

end#module
