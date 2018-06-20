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
#make lht and zt part of ARCHModel?
module ARCH
using Reexport
@reexport using StatsBase
using StatsFuns: normcdf, normccdf, normlogpdf, log2π, RFunctions.tdistrand
using Optim
using ForwardDiff
using Distributions
using Roots
using Compat #for circular_buffer
@static if Pkg.installed("StatsBase") >= v"0.22"
    import StatsBase: stderror
else
    import StatsBase: stderr
    const stderror = stderr
end
include("circular_buffer.jl")# no bounds checks
#using DataStructures: CircularBuffer
import Base: show, showerror, Random.rand, eltype, mean
import StatsBase: StatisticalModel, loglikelihood, nobs, fit, fit!, adjr2, aic,
                  bic, aicc, dof, coef, coefnames, coeftable, CoefTable
export ARCHModel, VolatilitySpec, simulate, selectmodel, StdNormal, StdTDist,
       Intercept, NoIntercept

abstract type VolatilitySpec{T} end
abstract type StandardizedDistribution{T} <: Distribution{Univariate, Continuous} end
abstract type MeanSpec{T} end

Base.@irrational sqrt2invpi 0.79788456080286535587 sqrt(big(2)/big(π))

struct NumParamError <: Exception
    expected::Int
    got::Int
end

struct LengthMismatchError <: Exception
    length1::Int
    length2::Int
end

function showerror(io::IO, e::NumParamError)
    print(io, "incorrect number of parameters: expected $(e.expected), got $(e.got).")
end

function showerror(io::IO, e::LengthMismatchError)
    print(io, "length of arrays does not match: $(e.length1) and $(e.length2).")
end

struct ARCHModel{T<:AbstractFloat,
                 VS<:VolatilitySpec,
                 SD<:StandardizedDistribution{T},
                 MS<:MeanSpec{T}
                 } <: StatisticalModel
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

function ARCHModel(spec::VS,
          data::Vector{T},
          ht::Vector{T},
          dist::SD,
          meanspec::MS
          ) where {T<:AbstractFloat,
                   VS<:VolatilitySpec,
                   SD<:StandardizedDistribution,
                   MS<:MeanSpec
                   }
    ARCHModel{T, VS, SD, MS}(spec, data, ht, dist, meanspec)
end

function ARCHModel(spec,
          data::Vector{T},
          ht::Vector{T} = zeros(data);
          dist=StdNormal{T}(),
          meanspec=NoIntercept{T}()
          ) where {T}
    AM = ARCHModel(spec, data, ht, dist, meanspec)
    loglik!(AM.ht, zeros(AM.ht), zeros(AM.ht), typeof(spec), typeof(dist), typeof(meanspec), AM.data,
            vcat(spec.coefs, dist.coefs, meanspec.coefs)
            )
    return AM
end

loglikelihood(am::ARCHModel) = loglik!(zeros(am.data), zeros(am.data), zeros(am.data),
                                       typeof(am.spec), typeof(am.dist),
                                       typeof(am.meanspec), am.data,
                                       vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs)
                                       )

nobs(am::ARCHModel) = length(am.data)
dof(am::ARCHModel) = nparams(typeof(am.spec)) + nparams(typeof(am.dist)) + nparams(typeof(am.meanspec))
coef(am::ARCHModel)=vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs)
coefnames(am::ARCHModel) = vcat(coefnames(typeof(am.spec)),
                                coefnames(typeof(am.dist)),
                                coefnames(typeof(am.meanspec))
                                )

function simulate(spec::VolatilitySpec{T}, nobs;
                  warmup=100,
                  dist::StandardizedDistribution{T}=StdNormal{T}(),
                  meanspec::MeanSpec{T}=NoIntercept{T}()
                  ) where {T<:AbstractFloat}
    data = zeros(T, nobs+warmup)
    ht = zeros(T, nobs+warmup)
    lht = zeros(T, nobs+warmup)
    zt = zeros(T, nobs+warmup)
    sim!(ht, lht, zt, data, spec, dist, meanspec)
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
@inline function bufloglik!(ht::AbstractVector{T2}, lht::AbstractVector{T2}, zt::AbstractVector{T2},
                    ::Type{VS}, ::Type{SD}, ::Type{MS},
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
                bufupdate!(ht, lht, zt, VS, MS, data, garchcoefs, meancoefs, t)
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

function bufloglik(spec::Type{VS}, dist::Type{SD}, meanspec::Type{MS},
                   data::Vector{<:AbstractFloat}, coefs::AbstractVector{T2}
                   ) where {VS<:VolatilitySpec, SD<:StandardizedDistribution,
                            MS<:MeanSpec, T2
                            }
    r = presample(VS)
    ht = CircularBuffer{T2}(r)
    lht = CircularBuffer{T2}(r)
    zt = CircularBuffer{T2}(r)
    bufloglik!(ht, lht, zt, spec, dist, meanspec, data, coefs)

end
function loglik!(ht::Vector{T2}, lht::Vector{T2}, zt::Vector{T2},
                 ::Type{VS}, ::Type{SD}, ::Type{MS},
                 data::Vector{<:AbstractFloat}, coefs::Vector{T2}
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
        ht[1:r] .= h0
        lht[1:r] .= log(h0)
        zt[1:r] .= (data[1:r].-mean(MS, meancoefs))./sqrt.(ht[1:r])
        LL = zero(T2)
        @fastmath for t = 1:T
            t > r && update!(ht, lht, zt, VS, MS, data, garchcoefs, meancoefs, t)
            zt[t] = (data[t]-mean(MS, meancoefs))/sqrt(ht[t])
            LL += -lht[t]/2 + logkernel(SD, zt[t], distcoefs)
        end#for
    end#inbounds
    LL += T*logconst(SD, distcoefs)
end#function

function logliks(spec, dist, meanspec, data, coefs::Vector{T}) where {T}
    garchcoefs, distcoefs, meancoefs = splitcoefs(coefs, spec, dist, meanspec)
    ht = zeros(T, length(data))
    lht = zeros(T, length(data))
    zt = zeros(T, length(data))
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

function sim!(ht::Vector{T1}, lht::Vector{T1}, zt::Vector{T1}, data::Vector{T1}, spec,
              dist::StandardizedDistribution{T1},
              meanspec::MeanSpec{T1}
              ) where {T1<:AbstractFloat}
    T =  length(data)
    r = presample(typeof(spec))
    @inbounds begin
        h0 = uncond(typeof(spec), spec.coefs)
        h0 > 0 || error("Model is nonstationary.")
        ht[1:r] .= h0
        lht[1:r] .= log(h0)
        rand!(dist, @view zt[1:r])
        data[1:r] .= sqrt(h0).*zt[1:r]
        data[1:r] .+= mean(typeof(meanspec), meanspec.coefs)
        @fastmath for t = r+1:T
            update!(ht, lht, zt, typeof(spec), typeof(meanspec), data, spec.coefs, meanspec.coefs, t)
            zt[t] = rand(dist)
            data[t] = mean(typeof(meanspec), meanspec.coefs) + sqrt(ht[t])*zt[t]
        end
    end
    return nothing
end

function fitbuf(::Type{VS}, ::Type{SD}, ::Type{MS},
              data::Vector{T}; algorithm=BFGS(), kwargs...
              ) where {VS<:VolatilitySpec, SD<:StandardizedDistribution, MS<:MeanSpec, T<:AbstractFloat}
    obj = x -> -bufloglik(VS, SD, MS, data, x)
    lowergarch, uppergarch = constraints(VS, T)
    lowerdist, upperdist = constraints(SD, T)
    lowermean, uppermean = constraints(MS, T)
    lower = vcat(lowergarch, lowerdist, lowermean)
    upper = vcat(uppergarch, upperdist, uppermean)
    coefs = vcat(startingvals(VS, data), startingvals(SD, data), startingvals(MS, data))
    res = optimize(obj, lower, upper, coefs, Fminbox(algorithm); kwargs...)
    return  Optim.minimizer(res)
end

function fit(::Type{VS}, data::Vector{T}; dist::Type{SD}=StdNormal{T}, meanspec::Type{MS}=Intercept{T},
    algorithm=BFGS(), kwargs...
    ) where {VS<:VolatilitySpec, SD<:StandardizedDistribution,
             MS<:MeanSpec, T<:AbstractFloat
             }
    ht = zeros(data)
    lht = zeros(data)
    zt = zeros(data)
    coefs = startingvals(VS, data)
    distcoefs = startingvals(SD, data)
    meancoefs = startingvals(MS, data)
    fit!(ht, lht, zt, coefs, distcoefs, meancoefs, VS, SD, MS, data; algorithm=algorithm, kwargs...)
    return ARCHModel(VS(coefs), data, ht, SD(distcoefs), MS(meancoefs))
end

function fit!(AM::ARCHModel; algorithm=BFGS(), kwargs...)
    AM.spec.coefs.=startingvals(typeof(AM.spec), AM.data)
    AM.dist.coefs.=startingvals(typeof(AM.dist), AM.data)
    AM.meanspec.coefs.=startingvals(typeof(AM.meanspec), AM.data)
    fit!(AM.ht, zeros(AM.ht), zeros(AM.ht), AM.spec.coefs, AM.dist.coefs, AM.meanspec.coefs, typeof(AM.spec),
         typeof(AM.dist), typeof(AM.meanspec), AM.data; algorithm=algorithm, kwargs...
         )
end

function fit!(ht::Vector{T}, lht::Vector{T}, zt::Vector{T}, garchcoefs::Vector{T}, distcoefs::Vector{T},
              meancoefs::Vector{T}, ::Type{VS}, ::Type{SD}, ::Type{MS},
              data::Vector{T}; algorithm=BFGS(), kwargs...
              ) where {VS<:VolatilitySpec, SD<:StandardizedDistribution, MS<:MeanSpec, T<:AbstractFloat}
    obj = x -> -loglik!(ht, lht, zt, VS, SD, MS, data, x)
    lowergarch, uppergarch = constraints(VS, T)
    lowerdist, upperdist = constraints(SD, T)
    lowermean, uppermean = constraints(MS, T)
    lower = vcat(lowergarch, lowerdist, lowermean)
    upper = vcat(uppergarch, upperdist, uppermean)
    coefs = vcat(garchcoefs, distcoefs, meancoefs)
    res = optimize(obj, lower, upper, coefs, Fminbox(algorithm); kwargs...)
    coefs .= res.minimizer
    ng = nparams(VS)
    ns = nparams(SD)
    nm = nparams(MS)
    garchcoefs .= coefs[1:ng]
    distcoefs .= coefs[ng+1:ng+ns]
    meancoefs .= coefs[ng+ns+1:ng+ns+nm]
    return nothing
end

function fit(AM::ARCHModel; algorithm=BFGS(), kwargs...)
    AM2=deepcopy(AM)
    fit!(AM2; algorithm=algorithm, kwargs...)
    return AM2
end


function selectmodel(::Type{VS}, data::Vector{T};
                     dist::Type{SD}=StdNormal{T}, meanspec::Type{MS}=Intercept{T},
                     maxpq=3, criterion=bic, show_trace=false, kwargs...
                     ) where {VS<:VolatilitySpec, T<:AbstractFloat,
                              SD<:StandardizedDistribution, MS<:MeanSpec
                              }
    ndims = my_unwrap_unionall(VS)-1#e.g., two (p and q) for GARCH{p, q, T}
    res = Array{ARCHModel, ndims}(ntuple(i->maxpq, ndims))
    Threads.@threads for ind in collect(CartesianRange(size(res)))
        res[ind] = fit(VS{ind.I...}, data; dist=dist, meanspec=meanspec)
    end
    for ind in collect(CartesianRange(size(res))) #seperate loop because juno crashes otherwise
        show_trace && println(split("$(VS{ind.I...})", ".")[2], " model has ",
                              uppercase(split("$criterion", ".")[2]), " ",
                              criterion(res[ind]), "."
                              )
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
    ccg, ccd, ccm = splitcoefs(cc, typeof(am.spec),
                               typeof(am.dist), typeof(am.meanspec)
                               )
    seg, sed, sem = splitcoefs(se, typeof(am.spec),
                               typeof(am.dist), typeof(am.meanspec)
                               )
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
include("EGARCH.jl")
end#module
