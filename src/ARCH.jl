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
#actually pass instances everywhere, at least for mean
#implement the remaining interface of StatisticalModel
#implement conditionalvariances/volas, stdresids
#use testsets
#remove circular_buffer.jl as soon as https://github.com/JuliaCollections/DataStructures.jl/pull/390 gets merged and tagged.
#make variance targeting an option?
# Float16/32 don't seem to work anymore. Problem in Optim?
module ARCH
using Reexport
@reexport using StatsBase
using StatsFuns: normcdf, normccdf, normlogpdf, log2π, RFunctions.tdistrand
using Optim
using ForwardDiff
using Distributions
using Roots
using Compat #for circular_buffer
import StatsBase: stderror
import DataStructures: CircularBuffer, _buffer_index_checked, _buffer_index,
                       capacity, isfull
include("circular_buffer.jl")# no bounds checks
import Base: show, showerror, Random.rand, eltype, mean
import StatsBase: StatisticalModel, loglikelihood, nobs, fit, fit!, adjr2, aic,
                  bic, aicc, dof, coef, coefnames, coeftable, CoefTable
export ARCHModel, VolatilitySpec, StandardizedDistribution, MeanSpec,
       simulate, selectmodel, StdNormal, StdTDist, Intercept, NoIntercept
"""
    VolatilitySpec{T}

Abstract supertype that volatility specifications inherit from.
"""
abstract type VolatilitySpec{T} end

"""
    StandardizedDistribution{T} <: Distributions.Distribution{Univariate, Continuous}

Abstract supertype that standardized distributions inherit from.
"""
abstract type StandardizedDistribution{T} <: Distribution{Univariate, Continuous} end

"""
    MeanSpec{T}
Abstract supertype that mean specifications inherit from.
"""
abstract type MeanSpec{T} end

Base.@irrational sqrt2invpi 0.79788456080286535587 sqrt(big(2)/big(π))

struct NumParamError <: Exception
    expected::Int
    got::Int
end

function showerror(io::IO, e::NumParamError)
    print(io, "incorrect number of parameters: expected $(e.expected), got $(e.got).")
end

"""
    ARCHModel(spec::VolatilitySpec, data::Vector, dist=StdNormal(),
	          meanspec=NoIntercept(), fitted=false
              )

Create an ARCHModel.

# Example:
```jldoctest
julia> ARCHModel(GARCH{1, 1}([1., .9, .05]), rand(10))

GARCH{1,1} model with Gaussian errors, T=10.


               ω  β₁   α₁
Parameters:  1.0 0.9 0.05
```
"""
mutable struct ARCHModel{T<:AbstractFloat,
                 VS<:VolatilitySpec,
                 SD<:StandardizedDistribution{T},
                 MS<:MeanSpec{T}
                 } <: StatisticalModel
    spec::VS
    data::Vector{T}
    dist::SD
    meanspec::MS
	fitted::Bool
    function ARCHModel{T, VS, SD, MS}(spec, data, dist, meanspec, fitted) where {T, VS, SD, MS}
        new(spec, data, dist, meanspec, fitted)
    end
end

function ARCHModel(spec::VS,
          data::Vector{T},
          dist::SD=StdNormal{T}(),
          meanspec::MS=NoIntercept{T}(),
		  fitted::Bool=false
          ) where {T<:AbstractFloat,
                   VS<:VolatilitySpec,
                   SD<:StandardizedDistribution,
                   MS<:MeanSpec
                   }
    ARCHModel{T, VS, SD, MS}(spec, data, dist, meanspec, fitted)
end

loglikelihood(am::ARCHModel) = loglik(typeof(am.spec), typeof(am.dist),
                                      typeof(am.meanspec), am.data,
                                      vcat(am.spec.coefs, am.dist.coefs,
                                           am.meanspec.coefs
                                           )
                                      )

nobs(am::ARCHModel) = length(am.data)
dof(am::ARCHModel) = nparams(typeof(am.spec)) + nparams(typeof(am.dist)) + nparams(typeof(am.meanspec))
coef(am::ARCHModel)=vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs)
coefnames(am::ARCHModel) = vcat(coefnames(typeof(am.spec)),
                                coefnames(typeof(am.dist)),
                                coefnames(typeof(am.meanspec))
                                )
isfitted(am::ARCHModel) = am.fitted

function simulate(spec::VolatilitySpec{T2}, nobs;
                  warmup=100,
                  dist::StandardizedDistribution{T2}=StdNormal{T2}(),
                  meanspec::MeanSpec{T2}=NoIntercept{T2}()
                  ) where {T2<:AbstractFloat}
    T = nobs+warmup
    data = zeros(T2, T)
    r = presample(typeof(spec))
    ht = CircularBuffer{T2}(r)
    lht = CircularBuffer{T2}(r)
    zt = CircularBuffer{T2}(r)
    @inbounds begin
        h0 = uncond(typeof(spec), spec.coefs)
        h0 > 0 || error("Model is nonstationary.")
        for t = 1:T
            if t>r
                update!(ht, lht, zt, typeof(spec), typeof(meanspec),
                        data, spec.coefs, meanspec.coefs, t
                        )
            else
                push!(ht, h0)
                push!(lht, log(h0))
            end
            push!(zt, rand(dist))
            data[t] = mean(typeof(meanspec), meanspec.coefs) + sqrt(ht[end])*zt[end]
        end
    end
    data[warmup+1:warmup+nobs]
end

@inline function splitcoefs(coefs, VS, SD, MS)
    ng = nparams(VS)
    nd = nparams(SD)
    nm = nparams(MS)
    length(coefs) == ng+nd+nm || throw(NumParamError(ng+nd+nm, length(coefs)))
    garchcoefs = coefs[1:ng]
    distcoefs = coefs[ng+1:ng+nd]
    meancoefs = coefs[ng+nd+1:ng+nd+nm]
    return garchcoefs, distcoefs, meancoefs
end

#this works on CircularBuffers. The idea is that ht/lht/zt need to be allocated
#inside of this function, when the type that Optim it with is known (because
#it calls it with dual numbers for autodiff to work). It works with arrays, too,
#but grows them by length(data); hence it should be called with an empty one-
#dimensional array of the right type.
@inline function loglik!(ht::AbstractVector{T2}, lht::AbstractVector{T2},
                         zt::AbstractVector{T2}, ::Type{VS}, ::Type{SD}, ::Type{MS},
                         data::Vector{<:AbstractFloat}, coefs::AbstractVector{T2}
                         ) where {VS<:VolatilitySpec, SD<:StandardizedDistribution,
                                  MS<:MeanSpec, T2
                                  }
    T = length(data)
    r = presample(VS)
    garchcoefs, distcoefs, meancoefs = splitcoefs(coefs, VS, SD, MS)
    T > r || error("Sample too small.")
    @inbounds begin
        h0 = uncond(VS, garchcoefs)
        h0 > 0 || return T2(NaN)
        LL = zero(T2)
        for t = 1:T
            if t > r
                update!(ht, lht, zt, VS, MS, data, garchcoefs, meancoefs, t)
            else
                push!(ht, h0)
                push!(lht, log(h0))
            end
            ht[end] < 0 && return T2(NaN)
            push!(zt, (data[t]-mean(MS, meancoefs))/sqrt(ht[end]))
            LL += -lht[end]/2 + logkernel(SD, zt[end], distcoefs)
        end#for
    end#inbounds
    LL += T*logconst(SD, distcoefs)
end#function

function loglik(spec::Type{VS}, dist::Type{SD}, meanspec::Type{MS},
                   data::Vector{<:AbstractFloat}, coefs::AbstractVector{T2}
                   ) where {VS<:VolatilitySpec, SD<:StandardizedDistribution,
                            MS<:MeanSpec, T2
                            }
    r = presample(VS)
    ht = CircularBuffer{T2}(r)
    lht = CircularBuffer{T2}(r)
    zt = CircularBuffer{T2}(r)
    loglik!(ht, lht, zt, spec, dist, meanspec, data, coefs)

end

function logliks(spec, dist, meanspec, data, coefs::Vector{T}) where {T}
    garchcoefs, distcoefs, meancoefs = splitcoefs(coefs, spec, dist, meanspec)
    ht = T[]
    lht = T[]
    zt = T[]
    loglik!(ht, lht, zt, spec, dist, meanspec, data, coefs)
    LLs = -lht./2+logkernel.(dist, zt, Ref{Vector{T}}(distcoefs)) + logconst(dist, distcoefs)
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


function fit!(garchcoefs::Vector{T}, distcoefs::Vector{T},
              meancoefs::Vector{T}, ::Type{VS}, ::Type{SD}, ::Type{MS},
              data::Vector{T}; algorithm=BFGS(), autodiff=:forward, kwargs...
              ) where {VS<:VolatilitySpec, SD<:StandardizedDistribution,
                       MS<:MeanSpec, T<:AbstractFloat
                       }
    obj = x -> -loglik(VS, SD, MS, data, x)
    lowergarch, uppergarch = constraints(VS, T)
    lowerdist, upperdist = constraints(SD, T)
    lowermean, uppermean = constraints(MS, T)
    lower = vcat(lowergarch, lowerdist, lowermean)
    upper = vcat(uppergarch, upperdist, uppermean)
    coefs = vcat(garchcoefs, distcoefs, meancoefs)
    res = optimize(obj, lower, upper, coefs, Fminbox(algorithm); autodiff=autodiff, kwargs...)
    coefs .= Optim.minimizer(res)
    ng = nparams(VS)
    ns = nparams(SD)
    nm = nparams(MS)
    garchcoefs .= coefs[1:ng]
    distcoefs .= coefs[ng+1:ng+ns]
    meancoefs .= coefs[ng+ns+1:ng+ns+nm]
    return nothing
end

function fit(::Type{VS}, data::Vector{T}; dist::Type{SD}=StdNormal{T},
             meanspec::Type{MS}=Intercept{T}, algorithm=BFGS(),
             autodiff=:forward, kwargs...
             ) where {VS<:VolatilitySpec, SD<:StandardizedDistribution,
                      MS<:MeanSpec, T<:AbstractFloat
                      }
    coefs = startingvals(VS, data)
    distcoefs = startingvals(SD, data)
    meancoefs = startingvals(MS, data)
    fit!(coefs, distcoefs, meancoefs, VS, SD, MS, data; algorithm=algorithm, autodiff=autodiff, kwargs...)
    return ARCHModel(VS(coefs), data, SD(distcoefs), MS(meancoefs), true)
end

function fit!(AM::ARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)
    AM.spec.coefs.=startingvals(typeof(AM.spec), AM.data)
    AM.dist.coefs.=startingvals(typeof(AM.dist), AM.data)
    AM.meanspec.coefs.=startingvals(typeof(AM.meanspec), AM.data)
    fit!(AM.spec.coefs, AM.dist.coefs, AM.meanspec.coefs, typeof(AM.spec),
         typeof(AM.dist), typeof(AM.meanspec), AM.data; algorithm=algorithm,
         autodiff=autodiff, kwargs...
         )
	AM.fitted=true
end


function fit(AM::ARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)
    AM2=deepcopy(AM)
    fit!(AM2; algorithm=algorithm, autodiff=autodiff, kwargs...)
    return AM2
end


function selectmodel(::Type{VS}, data::Vector{T};
                     dist::Type{SD}=StdNormal{T}, meanspec::Type{MS}=Intercept{T},
                     maxlags=3, criterion=bic, show_trace=false, kwargs...
                     ) where {VS<:VolatilitySpec, T<:AbstractFloat,
                              SD<:StandardizedDistribution, MS<:MeanSpec
                              }
    mylock=Threads.SpinLock()
    ndims = my_unwrap_unionall(VS)-1#e.g., two (p and q) for GARCH{p, q, T}
    res = Array{ARCHModel, ndims}(ntuple(i->maxlags, ndims))
    Threads.@threads for ind in collect(CartesianRange(size(res)))
        res[ind] = fit(VS{ind.I...}, data; dist=dist, meanspec=meanspec)
        if show_trace
            lock(mylock)
            Core.println(modname(VS{ind.I...}), " model has ",
                              uppercase(split("$criterion", ".")[2]), " ",
                              criterion(res[ind]), "."
                              )
            unlock(mylock)
        end
    end
    crits = criterion.(res)
    _, ind = findmin(crits)
    return res[ind]
end

function fit(::Type{SD}, data::Vector{T};
             algorithm=BFGS(), kwargs...
             ) where {SD<:StandardizedDistribution, T<:AbstractFloat}
    nparams(SD) == 0 && return SD{T}()
    obj = x -> -loglik(SD, data, x)
    lower, upper = constraints(SD, T)
    x0 = startingvals(SD, data)
    res = optimize(obj, lower, upper, x0, Fminbox(algorithm); kwargs...)
    coefs = res.minimizer
    return SD(coefs...)
end


function loglik(::Type{SD}, data::Vector{<:AbstractFloat},
                coefs::Vector{T2}
                ) where {SD<:StandardizedDistribution, T2}
    T = length(data)
    length(coefs) == nparams(SD) || throw(NumParamError(nparams(SD), length(coefs)))
    @inbounds begin
        LL = zero(T2)
        @fastmath for t = 1:T
            LL += logkernel(SD, data[t], coefs)
        end#for
    end#inbounds
    LL += T*logconst(SD, coefs)
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
    ccg, ccd, ccm = splitcoefs(cc, typeof(am.spec),
                               typeof(am.dist), typeof(am.meanspec)
                               )
    seg, sed, sem = splitcoefs(se, typeof(am.spec),
                               typeof(am.dist), typeof(am.meanspec)
                               )
    zzg = ccg ./ seg
    zzd = ccd ./ sed
    zzm = ccm ./ sem
	if isfitted(am)
	    println(io, "\n", modname(typeof(am.spec)), " model with ",
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

   else
	   println(io, "\n", modname(typeof(am.spec)), " model with ",
			   distname(typeof(am.dist)), " errors, T=", nobs(am), ".\n\n")
	   println(io, CoefTable(coef(am), coefnames(am), ["Parameters:"]))
   end
end

#from here https://stackoverflow.com/questions/46671965/printing-variable-subscripts-in-julia
subscript(i::Integer) = i<0 ? error("$i is negative") : join('₀'+d for d in reverse(digits(i)))

function modname(::Type{VS}) where VS<:VolatilitySpec
    s = split("$(VS)", ".")[2]
    s = s[1:findlast(s, ',')-1] * '}'
end

include("meanspecs.jl")
include("standardizeddistributions.jl")
include("GARCH.jl")
include("EGARCH.jl")
end#module
