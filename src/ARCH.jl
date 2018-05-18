__precompile__()
#Todo:
#docs
#plotting via timeseries
#marketdata
#PkgBenchmark
#HAC s.e.s from CovariancesMatrices.jl?
#loglik! etc should take distcoefs seperately
#demean?
#GARCH instances should carry params like distributions do (eg for simulate, but not for loglike, b/c of ForwardDiff), but then it needs to be parameterizedon float type
#bic table
#simulate ARCHModel
#how to export arch?
#what should simulate return?


module ARCH
using Reexport
@reexport using StatsBase
using StatsFuns: normccdf, normlogpdf, log2π, RFunctions.tdistrand
using Optim
using ForwardDiff
using Distributions
using Roots

import Base: show, showerror, Random.rand, eltype
import StatsBase: StatisticalModel, loglikelihood, nobs, fit, fit!, adjr2, aic, bic, aicc, dof, coef, coefnames, coeftable, CoefTable, stderr
#export            StatisticalModel, loglikelihood, nobs, fit, fit!, adjr2, aic, bic, aicc, dof, coef, coefnames, coeftable, CoefTable, stderr
export ARCHModel, VolatilitySpec, simulate, selectmodel, StdNormal, StdTDist

abstract type VolatilitySpec end
abstract type StandardizedDistribution{T} <: Distribution{Univariate, Continuous}
end


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

struct ARCHModel{VS<:VolatilitySpec, T<:AbstractFloat, SD<:StandardizedDistribution{T}} <: StatisticalModel
    data::Vector{T}
    ht::Vector{T}
    coefs::Vector{T}
    dist::SD

    function ARCHModel{VS, T, SD}(data, ht, coefs, dist) where {VS, T, SD}
        length(coefs) == nparams(VS)  || throw(NumParamError(nparams(VS), length(coefs)))
        length(data) == length(ht)  || throw(LengthMismatchError(length(data), length(ht)))
        new(copy(data), copy(ht), copy(coefs), dist)
    end
end
ARCHModel(::Type{VS}, data::Vector{T}, ht::Vector{T}, coefs::Vector{T}, dist::SD=StdNormal{T}()) where {VS<:VolatilitySpec, T, SD<:StandardizedDistribution{T}} = ARCHModel{VS, T, SD}(data, ht, coefs, dist)
ARCHModel(::Type{VS}, data::Vector{T}, coefs::Vector{T}, dist::SD=StdNormal{T}()) where {VS<:VolatilitySpec, T, SD<:StandardizedDistribution{T}} = (ht = zeros(data); loglik!(ht, VS, SD,  data, vcat(coefs, dist.coefs)); ARCHModel(VS, data, ht, coefs, dist))

loglikelihood(am::ARCHModel{VS}) where {VS<:VolatilitySpec} = loglik!(zeros(am.data), VS, typeof(am.dist), am.data, vcat(am.coefs, am.dist.coefs))
nobs(am::ARCHModel) = length(am.data)
dof(am::ARCHModel{VS}) where {VS<:VolatilitySpec} = nparams(VS) + nparams(typeof(am.dist))
coef(am::ARCHModel)=vcat(am.coefs, am.dist.coefs)
coefnames(am::ARCHModel{VS}) where {VS<:VolatilitySpec} = vcat(coefnames(VS), coefnames(typeof(am.dist)))

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
    length(coefs) == ng+nd || throw(NumParamError(ng+nd, length(coefs)))
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
    f = x -> ARCH.logliks(VS, typeof(am.dist), am.data, x)
    g = x -> sum(ARCH.logliks(VS, typeof(am.dist), am.data, x))
    J = ForwardDiff.jacobian(f, vcat(am.coefs, am.dist.coefs))
    V = J'J #outer product of scores
    H = ForwardDiff.hessian(g, vcat(am.coefs, am.dist.coefs))
    Ji = try
        -inv(H) #inverse of observed Fisher information. Note: B&W use expected information.
    catch e
        if e isa LinAlg.SingularException
            warn("Fisher information is singular; standard errors are inaccurate.")
            -pinv(H)
        else
            rethrow(e)
        end
    end
    v = diag(Ji*V*Ji)
    any(v.<0) && warn("negative variance encountered; standard errors are inaccurate.")
    return sqrt.(abs.(v)) #Huber sandwich
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

fit(::Type{VS}, data::Vector{T}, ::Type{SD}=StdNormal{T},  algorithm=BFGS; kwargs...) where {VS<:VolatilitySpec, SD<:StandardizedDistribution, T<:AbstractFloat} = (ht = zeros(data); coefs=startingvals(VS, data); distcoefs=startingvals(SD, data); fit!(ht, coefs, distcoefs, VS, SD, data, algorithm; kwargs...); return ARCHModel(VS, data, ht, coefs, SD(distcoefs...)))
fit!(AM::ARCHModel{VS}, algorithm=BFGS; kwargs...) where {VS<:VolatilitySpec} = (AM.coefs.=startingvals(VS, AM.data); AM.dist.coefs.=startingvals(typeof(AM.dist), AM.data); fit!(AM.ht, AM.coefs, AM.dist.coefs, VS, typeof(AM.dist), AM.data, algorithm; kwargs...))
fit(AM::ARCHModel{VS}, algorithm=BFGS; kwargs...) where {VS<:VolatilitySpec} = (AM2=deepcopy(AM); fit!(AM2, algorithm=BFGS; kwargs...); return AM2)


function selectmodel(::Type{VS}, data::Vector{T}, dist::Type{SD}=StdNormal{T}, maxpq=3; criterion=bic, show_trace=false, kwargs...) where {VS<:VolatilitySpec, T<:AbstractFloat, SD<:StandardizedDistribution}
    ndims = my_unwrap_unionall(VS)#e.g., two (p and q) for GARCH{p, q}
    res = Array{ARCHModel, ndims}(ntuple(i->maxpq, ndims))
    Threads.@threads for ind in collect(CartesianRange(size(res)))
        res[ind] = fit(VS{ind.I...}, data, dist)
    end
    for ind in collect(CartesianRange(size(res))) #seperate loop because juno crashes otherwise
        show_trace && println(split("$(VS{ind.I...})", ".")[2], " model has ", uppercase(split("$criterion", ".")[2]), " ", criterion(res[ind]), ".")
    end
    crits = criterion.(res)
    _, ind = findmin(crits)
    return res[ind]
end

function fit(::Type{SD}, data::Vector{T}, algorithm=BFGS; kwargs...) where {SD<:StandardizedDistribution, T<:AbstractFloat}
    nparams(SD) == 0 && return SD{T}()
    obj = x -> -loglik(SD, data, x)
    lower, upper = constraints(SD, T)
    x0 = startingvals(SD, data)
    res = optimize(obj, x0, lower, upper, Fminbox{algorithm}(); kwargs...)
    coefs = res.minimizer
    return SD(coefs...)
end


function loglik(::Type{SD}, data::Vector{<:AbstractFloat}, coefs::Vector{T2}) where {SD<:StandardizedDistribution, T2}
    T = length(data)
    length(coefs) == nparams(SD) || throw(NumParamError(nparams(SD), length(coefs)))
    @inbounds begin
        LL = zero(T2)
        @fastmath for t = 1:T
            LL += logkernel(SD, data[t], coefs)
        end#for
    end#inbounds
    LL +=  T*logconst(SD, coefs)
end#function

Base.eltype(::StandardizedDistribution{T})  where {T} = T

#count the number of type vars. there's probably a better way.
function my_unwrap_unionall(a::ANY)
    count = 0
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
    println(io, "\n", split("$VS", ".")[2], " model with ", distname(typeof(am.dist)), " errors, T=",
        nobs(am), ".\n\n", coeftable(am))
end

#from here https://stackoverflow.com/questions/46671965/printing-variable-subscripts-in-julia
subscript(i::Integer) = i<0 ? error("$i is negative") : join('₀'+d for d in reverse(digits(i)))

include("standardizeddistributions.jl")
include("GARCH.jl")

end#module
