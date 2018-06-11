__precompile__()
#Todo:
#docs
#plotting via timeseries
#marketdata
#PkgBenchmark
#HAC s.e.s from CovariancesMatrices.jl?
#simulate ARCHModel
#how to export arch?
#what should simulate return?
#sim should take data 2nd
#actually pass instances everywhere, at least for mean

module ARCH
using Reexport
@reexport using StatsBase
using StatsFuns: normccdf, normlogpdf, log2π, RFunctions.tdistrand
using Optim
using ForwardDiff
using Distributions
using Roots

@static if Pkg.installed("StatsBase") >= v"0.22"
    import StatsBase: stderror
else
    import StatsBase: stderr
    const stderror = stderr
end

import Base: show, showerror, Random.rand, eltype, mean
import StatsBase: StatisticalModel, loglikelihood, nobs, fit, fit!, adjr2, aic, bic, aicc, dof, coef, coefnames, coeftable, CoefTable
export ARCHModel, VolatilitySpec, simulate, selectmodel, StdNormal, StdTDist, Intercept, NoIntercept

abstract type VolatilitySpec{T} end
abstract type StandardizedDistribution{T} <: Distribution{Univariate, Continuous}
end
abstract type MeanSpec{T} end


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

struct ARCHModel{T<:AbstractFloat, VS<:VolatilitySpec, SD<:StandardizedDistribution{T}, MS<:MeanSpec{T}} <: StatisticalModel
    spec::VS
    data::Vector{T}
    ht::Vector{T}
    dist::SD
    meanspec::MS
    function ARCHModel{T, VS, SD, MS}(spec, data, ht, dist, meanspec) where {T, VS, SD, MS}
        length(data) == length(ht)  || throw(LengthMismatchError(length(data), length(ht)))
        new(deepcopy(spec), copy(data), copy(ht), dist, meanspec)
    end
end
ARCHModel(spec::VS, data::Vector{T}, ht::Vector{T}, dist::SD, meanspec::MS) where {T<:AbstractFloat, VS<:VolatilitySpec, SD<:StandardizedDistribution, MS<:MeanSpec}  = ARCHModel{T, VS, SD, MS}(spec, data, ht, dist, meanspec)
ARCHModel(spec, data::Vector{T}, ht::Vector{T} = zeros(data); dist=StdNormal{T}(), meanspec=NoIntercept{T}()) where {T} = ( AM = ARCHModel(spec, data, ht, dist, meanspec); loglik!(AM.ht, typeof(spec), typeof(dist), typeof(meanspec), AM.data, vcat(spec.coefs, dist.coefs, meanspec.coefs)); AM)

loglikelihood(am::ARCHModel) = loglik!(zeros(am.data), typeof(am.spec), typeof(am.dist), typeof(am.meanspec), am.data, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
nobs(am::ARCHModel) = length(am.data)
dof(am::ARCHModel) = nparams(typeof(am.spec)) + nparams(typeof(am.dist)) + nparams(typeof(am.meanspec))
coef(am::ARCHModel)=vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs)
coefnames(am::ARCHModel) = vcat(coefnames(typeof(am.spec)), coefnames(typeof(am.dist)), coefnames(typeof(am.meanspec)))

function simulate(spec::VolatilitySpec{T}, nobs; dist::StandardizedDistribution{T}=StdNormal{T}(), meanspec::MeanSpec{T}=Intercept{T}()) where {T<:AbstractFloat}
    const warmup = 100
    data = zeros(T, nobs+warmup)
    ht = zeros(T, nobs+warmup)
    sim!(ht, spec, dist, meanspec, data)
    data[warmup+1:warmup+nobs]
end
function splitcoefs(coefs, VS, SD, MS)
    ng = nparams(VS)
    nd = nparams(SD)
    nm = nparams(MS)
    length(coefs) == ng+nd+nm || throw(NumParamError(ng+nd+nm, length(coefs)))
    garchcoefs = coefs[1:ng]
    distcoefs = coefs[ng+1:ng+nd]
    meancoefs = coefs[ng+nd+1:ng+nd+nm]
    return garchcoefs, distcoefs, meancoefs
end
function loglik!(ht::Vector{T2}, ::Type{VS}, ::Type{SD}, ::Type{MS}, data::Vector{<:AbstractFloat}, coefs::Vector{T2}) where {VS<:VolatilitySpec, SD<:StandardizedDistribution, MS<:MeanSpec, T2}
    T = length(data)
    r = presample(VS)
    garchcoefs, distcoefs, meancoefs = splitcoefs(coefs, VS, SD, MS)
    T > r || error("Sample too small.")
    @inbounds begin
        h0 = uncond(VS, garchcoefs)
        h0 > 0 || return T2(NaN)
        lh0 = log(h0)
        ht[1:r] .= h0
        LL = zero(T2)
        @fastmath for t = 1:T
            t > r && update!(ht, VS, MS, data, garchcoefs, meancoefs, t)
            LL += -log(ht[t])/2 + logkernel(SD, (data[t]-mean(MS, meancoefs))/sqrt(ht[t]), distcoefs)
        end#for
    end#inbounds
    LL += T*logconst(SD, distcoefs)
end#function

function logliks(spec, dist, meanspec, data, coefs::Vector{T}) where {T}
    garchcoefs, distcoefs, meancoefs = splitcoefs(coefs, spec, dist, meanspec)
    ht = zeros(T, length(data))
    loglik!(ht, spec, dist, meanspec, data, coefs)
    LLs = -log.(ht)/2+logkernel.(dist, (data-mean(meanspec, meancoefs))./sqrt.(ht), Ref{Vector{T}}(distcoefs))+logconst(dist, distcoefs)
end

function stderror(am::ARCHModel)
    f = x -> ARCH.logliks(typeof(am.spec), typeof(am.dist), typeof(am.meanspec), am.data, x)
    g = x -> sum(ARCH.logliks(typeof(am.spec), typeof(am.dist), typeof(am.meanspec), am.data, x))
    J = ForwardDiff.jacobian(f, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
    V = J'J #outer product of scores
    H = ForwardDiff.hessian(g, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
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

function sim!(ht::Vector{T1}, spec, dist::StandardizedDistribution{T1}, meanspec::MeanSpec{T1}, data::Vector{T1}) where {T1<:AbstractFloat}
    T =  length(data)
    r = presample(typeof(spec))
    @inbounds begin
        h0 = uncond(typeof(spec), spec.coefs)
        h0 > 0 || error("Model is nonstationary.")
        rand!(dist, @view data[1:r])
        data[1:r] .*= sqrt(h0)
        data[1:r] .+= mean(typeof(meanspec), meanspec.coefs)
        @fastmath for t = r+1:T
            update!(ht, typeof(spec), typeof(meanspec), data, spec.coefs, meanspec.coefs, t)
            data[t] = mean(typeof(meanspec), meanspec.coefs) + sqrt(ht[t])*rand(dist)
        end
    end
    return nothing
end

function fit!(ht::Vector{T}, garchcoefs::Vector{T}, distcoefs::Vector{T}, meancoefs::Vector{T}, ::Type{VS}, ::Type{SD}, ::Type{MS}, data::Vector{T}, algorithm=BFGS; kwargs...) where {VS<:VolatilitySpec, SD<:StandardizedDistribution, MS<:MeanSpec, T<:AbstractFloat}
    obj = x -> -loglik!(ht, VS, SD, MS, data, x)
    lowergarch, uppergarch = constraints(VS, T)
    lowerdist, upperdist = constraints(SD, T)
    lowermean, uppermean = constraints(MS, T)
    lower = vcat(lowergarch, lowerdist, lowermean)
    upper = vcat(uppergarch, upperdist, uppermean)
    coefs = vcat(garchcoefs, distcoefs, meancoefs)
    res = optimize(obj, coefs, lower, upper, Fminbox{algorithm}(); kwargs...)
    coefs .= res.minimizer
    ng = nparams(VS)
    ns = nparams(SD)
    nm = nparams(MS)
    garchcoefs .= coefs[1:ng]
    distcoefs .= coefs[ng+1:ng+ns]
    meancoefs .= coefs[ng+ns+1:ng+ns+nm]
    return nothing
end

fit(::Type{VS}, data::Vector{T}; dist::Type{SD}=StdNormal{T}, meanspec::Type{MS}=Intercept{T}, algorithm=BFGS, kwargs...) where {VS<:VolatilitySpec, SD<:StandardizedDistribution, MS<:MeanSpec, T<:AbstractFloat} = (ht = zeros(data); coefs = startingvals(VS, data); distcoefs = startingvals(SD, data); meancoefs = startingvals(MS, data); fit!(ht, coefs, distcoefs, meancoefs, VS, SD, MS, data, algorithm; kwargs...); return ARCHModel(VS(coefs), data, ht, SD(distcoefs...), MS(meancoefs...)))
fit!(AM::ARCHModel; algorithm=BFGS, kwargs...) = (AM.spec.coefs.=startingvals(typeof(AM.spec), AM.data); AM.dist.coefs.=startingvals(typeof(AM.dist), AM.data); AM.meanspec.coefs.=startingvals(typeof(AM.meanspec), AM.data); fit!(AM.ht, AM.spec.coefs, AM.dist.coefs, AM.meanspec.coefs, typeof(AM.spec), typeof(AM.dist), typeof(AM.meanspec), AM.data; algorithm=algorithm, kwargs...))
fit(AM::ARCHModel; algorithm=BFGS, kwargs...) = (AM2=deepcopy(AM); fit!(AM2; algorithm=algorithm, kwargs...); return AM2)


function selectmodel(::Type{VS}, data::Vector{T}; dist::Type{SD}=StdNormal{T}, meanspec::Type{MS}=Intercept{T}, maxpq=3, criterion=bic, show_trace=false, kwargs...) where {VS<:VolatilitySpec, T<:AbstractFloat, SD<:StandardizedDistribution, MS<:MeanSpec}
    ndims = my_unwrap_unionall(VS)-1#e.g., two (p and q) for GARCH{p, q, T}
    res = Array{ARCHModel, ndims}(ntuple(i->maxpq, ndims))
    Threads.@threads for ind in collect(CartesianRange(size(res)))
        res[ind] = fit(VS{ind.I...}, data; dist=dist, meanspec=meanspec)
    end
    for ind in collect(CartesianRange(size(res))) #seperate loop because juno crashes otherwise
        show_trace && println(split("$(VS{ind.I...})", ".")[2], " model has ", uppercase(split("$criterion", ".")[2]), " ", criterion(res[ind]), ".")
    end
    crits = criterion.(res)
    _, ind = findmin(crits)
    return res[ind]
end

function fit(::Type{SD}, data::Vector{T}; algorithm=BFGS, kwargs...) where {SD<:StandardizedDistribution, T<:AbstractFloat}
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

Base.eltype(::StandardizedDistribution{T}) where {T} = T

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
    se = stderror(am)
    zz = cc ./ se
    CoefTable(hcat(cc, se, zz, 2.0 * normccdf.(abs.(zz))),
              ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
              coefnames(am), 4)
end

function show(io::IO, am::ARCHModel)
    cc = coef(am)
    se = stderror(am)
    ccg, ccd, ccm = splitcoefs(cc, typeof(am.spec), typeof(am.dist), typeof(am.meanspec))
    seg, sed, sem = splitcoefs(se, typeof(am.spec), typeof(am.dist), typeof(am.meanspec))
    zzg = ccg ./ seg
    zzd = ccd ./ sed
    zzm = ccm ./ sem
    println(io, "\n", split("$(typeof(am.spec))", ".")[2], " model with ",
            distname(typeof(am.dist)), " errors, T=", nobs(am), ".\n\n")

    length(sem) > 0 && println(io, "Mean equation parameters:", "\n\n",
                               CoefTable(hcat(ccm, sem, zzm, 2.0 * normccdf.(abs.(zzm))),
                                         ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
                                         coefnames(typeof(am.meanspec)), 4
                                         )
                              )
    println(io, "Volatility parameters:", "\n\n",
            CoefTable(hcat(ccg, seg, zzg, 2.0 * normccdf.(abs.(zzg))),
                      ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
                      coefnames(typeof(am.spec)), 4
                      )
            )
    length(sed) > 0 && println(io, "Distribution parameters:", "\n\n",
                               CoefTable(hcat(ccd, sed, zzd, 2.0 * normccdf.(abs.(zzd))),
                                         ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
                                         coefnames(typeof(am.dist)), 4
                                         )
                              )
end

#from here https://stackoverflow.com/questions/46671965/printing-variable-subscripts-in-julia
subscript(i::Integer) = i<0 ? error("$i is negative") : join('₀'+d for d in reverse(digits(i)))

include("meanspecs.jl")
include("standardizeddistributions.jl")
include("GARCH.jl")

end#module
