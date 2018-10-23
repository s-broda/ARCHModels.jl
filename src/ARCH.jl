#Todo:
#plotting via timeseries
#PkgBenchmark
#HAC s.e.s from CovariancesMatrices.jl?
#how to export arch?
#Forecasting
#actually pass instances everywhere, at least for mean
#Float16/32 don't seem to work anymore. Problem in Optim?
#support missing data? timeseries?
#a simulated AM should probably contain a (zero) intercept, so that fit! is consistent with fit.
#the constructor for ARCHModel should make a copy of its args
#implement lrtest
#I've observed non-deterministic segfaults in testing selectmodel with threading enabled. seems to happen only in 1.0.1, and only locally, not on CI.  Investigate!
#allow uninititalized constructors for VolatilitySpec, MeanSpec and StandardizedDistribution? If so, then be consistent with how they are defined
#  (change for meanspec and dist ), document, and test. Also, NaN is prob. safer than undef.
#constructors for meanspec, distributions should check length of coef vector
#rename to ARCHModels
#mean(meanspec) should take an instance.
#allow arbitrary distributions by making a wrapper type Standardized{<:UnivariateContinuousDistribution}?
"""
The ARCH package for Julia. For documentation, see https://s-broda.github.io/ARCH.jl/latest.
"""
module ARCH
using Reexport
@reexport using StatsBase
using StatsFuns: normcdf, normccdf, normlogpdf, log2π, RFunctions.tdistrand
using SpecialFunctions: beta, lgamma
using Optim
using ForwardDiff
using Distributions
using Roots
using LinearAlgebra
using DataStructures: CircularBuffer
using DelimitedFiles
import Base: show, showerror, eltype
import Statistics: mean
import Random: rand
import StatsBase: StatisticalModel, stderror, loglikelihood, nobs, fit, fit!, confint, aic,
                  bic, aicc, dof, coef, coefnames, coeftable, CoefTable,
				  informationmatrix, islinear, score, vcov, residuals

export ARCHModel, VolatilitySpec, StandardizedDistribution, MeanSpec,
       simulate, simulate!, selectmodel, StdNormal, StdTDist, Intercept,
       NoIntercept, BG96, volatilities, mean

"""
    BG96
Data from [Bollerslev and Ghysels (JBES 1996)](https://doi.org/10.2307/1392425).
"""
const BG96 = readdlm(joinpath(dirname(pathof(ARCH)), "data", "bollerslev_ghysels.txt"), skipstart=1)[:, 1];

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
    ARCHModel{T<:AbstractFloat,
              VS<:VolatilitySpec,
              SD<:StandardizedDistribution{T},
              MS<:MeanSpec{T}
              } <: StatisticalModel
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

"""
    ARCHModel(spec::VolatilitySpec, data::Vector; dist=StdNormal(),
	          meanspec=NoIntercept(), fitted=false
              )

Create an ARCHModel.

# Example:
```jldoctest
julia> ARCHModel(GARCH{1, 1}([1., .9, .05]), randn(10))

GARCH{1,1} model with Gaussian errors, T=10.


                             ω  β₁   α₁
Volatility parameters:     1.0 0.9 0.05
```
"""
function ARCHModel(spec::VS,
          data::Vector{T};
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
islinear(am::ARCHModel) = false

function confint(am::ARCHModel, level::Real=0.95)
    hcat(coef(am), coef(am)) .+ stderror(am)*quantile(Normal(),(1. -level)/2.)*[1. -1.]
end

isfitted(am::ARCHModel) = am.fitted

"""
    simulate(am::ARCHModel; warmup=100)
    simulate(spec::VolatilitySpec, nobs; warmup=100, dist=StdNormal(), meanspec=NoIntercept())
Simulate an ARCHModel.
"""
function simulate end

function simulate(am::ARCHModel; warmup=100)
	am2 = deepcopy(am)
    simulate(am2.spec, nobs(am2); warmup=warmup, dist=am2.dist, meanspec=am2.meanspec)
end

function simulate(spec::VolatilitySpec{T2}, nobs; warmup=100, dist::StandardizedDistribution{T2}=StdNormal{T2}(),
                  meanspec::MeanSpec{T2}=NoIntercept{T2}()
                  ) where {T2<:AbstractFloat}
    data = zeros(T2, nobs)
    _simulate!(data,  spec; warmup=warmup, dist=dist, meanspec=meanspec)
    ARCHModel(spec, data; dist=dist, meanspec=meanspec, fitted=false)
end

"""
    simulate!(am::ARCHModel; warmup=100)
Simulate an ARCHModel, modifying `am` in place.
"""
function simulate!(am::ARCHModel; warmup=100)
	am.fitted = false
    _simulate!(am.data, am.spec; warmup=warmup, dist=am.dist, meanspec=am.meanspec)
    am
end

function _simulate!(data::Vector{T2}, spec::VolatilitySpec{T2};
                  warmup=100,
                  dist::StandardizedDistribution{T2}=StdNormal{T2}(),
                  meanspec::MeanSpec{T2}=NoIntercept{T2}()
                  ) where {T2<:AbstractFloat}
	@assert warmup>0
	append!(data, zeros(T2, warmup))
    T = length(data)
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
    deleteat!(data, 1:warmup)
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
"""
    volatilities(am::ARCHModel)
Return the conditional volatilities.
"""
function volatilities(am::ARCHModel{T, VS, SD, MS}) where {T, VS, SD, MS}
	ht = Vector{T}(undef, 0)
	lht = Vector{T}(undef, 0)
	zt = Vector{T}(undef, 0)
	loglik!(ht, lht, zt, VS, SD, MS, am.data, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
	return sqrt.(ht)
end

"""
    residuals(am::ARCHModel; standardized=true)
Return the residuals of the model. Pass `standardized=false` for the non-devolatized residuals.
"""
function residuals(am::ARCHModel{T, VS, SD, MS}; standardized=true) where {T, VS, SD, MS}
	if standardized
		ht = Vector{T}(undef, 0)
		lht = Vector{T}(undef, 0)
		zt = Vector{T}(undef, 0)
		loglik!(ht, lht, zt, VS, SD, MS, am.data, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
		return zt
	else
		return am.data.-mean(MS, am.meanspec.coefs)
	end
end

#this works on CircularBuffers. The idea is that ht/lht/zt need to be allocated
#inside of this function, when the type that Optim it with is known (because
#it calls it with dual numbers for autodiff to work). It works with arrays, too,
#but grows them by length(data); hence it should be called with an empty one-
#dimensional array of the right type.
@inline function loglik!(ht::AbstractVector{T2}, lht::AbstractVector{T2},
                         zt::AbstractVector{T2}, ::Type{VS}, ::Type{SD}, ::Type{MS},
                         data::Vector{T1}, coefs::AbstractVector{T2}
                         ) where {VS<:VolatilitySpec, SD<:StandardizedDistribution,
                                  MS<:MeanSpec, T1<:AbstractFloat, T2
                                  }
    garchcoefs, distcoefs, meancoefs = splitcoefs(coefs, VS, SD, MS)
    #the below 6 lines can be removed when using Fminbox
    lowergarch, uppergarch = constraints(VS, T1)
    lowerdist, upperdist = constraints(SD, T1)
    lowermean, uppermean = constraints(MS, T1)
    lower = vcat(lowergarch, lowerdist, lowermean)
    upper = vcat(uppergarch, upperdist, uppermean)
    all(lower.<coefs.<upper) || return T2(-Inf)
    T = length(data)
    r = presample(VS)
    T > r || error("Sample too small.")
    @inbounds begin
        h0 = var(data) # could be moved outside
        #h0 = uncond(VS, garchcoefs)
        #h0 > 0 || return T2(NaN)
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
    LLs = -lht./2 .+ logkernel.(dist, zt, Ref{Vector{T}}(distcoefs)) .+ logconst(dist, distcoefs)
end

function informationmatrix(am::ARCHModel; expected::Bool=true)
	expected && error("expected informationmatrix is not implemented for ARCHModel. Use expected=false.")
	g = x -> sum(ARCH.logliks(typeof(am.spec), typeof(am.dist), typeof(am.meanspec), am.data, x))
	H = ForwardDiff.hessian(g, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
	J = -H/nobs(am)
end

function scores(am::ARCHModel)
	f = x -> ARCH.logliks(typeof(am.spec), typeof(am.dist), typeof(am.meanspec), am.data, x)
	S = ForwardDiff.jacobian(f, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
end

score(am::ARCHModel) = sum(scores(am), dims=1)


function vcov(am::ARCHModel)
	S = scores(am)
    V = S'S/nobs(am)
    J = informationmatrix(am; expected=false) #Note: B&W use expected information.
    Ji = try
        inv(J)
    catch e
        if e isa LinearAlgebra.SingularException
            @warn "Fisher information is singular; vcov matrix is inaccurate."
            pinv(J)
        else
            rethrow(e)
        end
    end
    v = Ji*V*Ji/nobs(am) #Huber sandwich
    all(diag(v).>0) || @warn "non-positive variance encountered; vcov matrix is inaccurate."
    v
end

stderror(am::ARCHModel) = sqrt.(abs.(diag(vcov(am))))

function _fit!(garchcoefs::Vector{T}, distcoefs::Vector{T},
              meancoefs::Vector{T}, ::Type{VS}, ::Type{SD}, ::Type{MS},
              data::Vector{T}; algorithm=BFGS(), autodiff=:forward, kwargs...
              ) where {VS<:VolatilitySpec, SD<:StandardizedDistribution,
                       MS<:MeanSpec, T<:AbstractFloat
                       }
    obj = x -> -loglik(VS, SD, MS, data, x)
    coefs = vcat(garchcoefs, distcoefs, meancoefs)
    #for fminbox:
    # lowergarch, uppergarch = constraints(VS, T)
    # lowerdist, upperdist = constraints(SD, T)
    # lowermean, uppermean = constraints(MS, T)
    # lower = vcat(lowergarch, lowerdist, lowermean)
    # upper = vcat(uppergarch, upperdist, uppermean)
    # res = optimize(obj, lower, upper, coefs, Fminbox(algorithm); autodiff=autodiff, kwargs...)
    res = optimize(obj, coefs, algorithm; autodiff=autodiff, kwargs...)
    coefs .= Optim.minimizer(res)
    ng = nparams(VS)
    ns = nparams(SD)
    nm = nparams(MS)
    garchcoefs .= coefs[1:ng]
    distcoefs .= coefs[ng+1:ng+ns]
    meancoefs .= coefs[ng+ns+1:ng+ns+nm]
    return nothing
end

"""
    fit(VS::Type{<:VolatilitySpec}, data; dist=StdNormal, meanspec=Intercept,
        algorithm=BFGS(), autodiff=:forward, kwargs...)

Fit the ARCH model specified by `VS` to data.

# Keyword arguments:
- `dist=StdNormal`: the error distribution.
- `meanspec=Intercept`: the mean specification.
- `algorithm=BFGS(), autodiff=:forward, kwargs...`: passed on to the optimizer.

# Example: EGARCH{1, 1, 1} model without intercept, Student's t errors.
```jldoctest
julia> fit(EGARCH{1, 1, 1}, BG96; meanspec=NoIntercept, dist=StdTDist)

EGARCH{1,1,1} model with Student's t errors, T=1974.


Volatility parameters:

       Estimate Std.Error   z value Pr(>|z|)
ω    -0.0162014 0.0186806 -0.867286   0.3858
γ₁   -0.0378454  0.018024  -2.09972   0.0358
β₁     0.977687  0.012558   77.8538   <1e-99
α₁     0.255804 0.0625497   4.08961    <1e-4

Distribution parameters:

     Estimate Std.Error z value Pr(>|z|)
ν     4.12423   0.40059 10.2954   <1e-24
```
"""
function fit(::Type{VS}, data::Vector{T}; dist::Type{SD}=StdNormal{T},
             meanspec::Type{MS}=Intercept{T}, algorithm=BFGS(),
             autodiff=:forward, kwargs...
             ) where {VS<:VolatilitySpec, SD<:StandardizedDistribution,
                      MS<:MeanSpec, T<:AbstractFloat
                      }
    coefs = startingvals(VS, data)
    distcoefs = startingvals(SD, data)
    meancoefs = startingvals(MS, data)
    _fit!(coefs, distcoefs, meancoefs, VS, SD, MS, data; algorithm=algorithm, autodiff=autodiff, kwargs...)
    return ARCHModel(VS(coefs), data; dist=SD(distcoefs), meanspec=MS(meancoefs), fitted=true)
end

"""
    fit!(am::ARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)

Fit the ARCHModel specified by `am`, modifying `am` in place. Keyword arguments
are passed on to the optimizer.
"""
function fit!(am::ARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)
    am.spec.coefs.=startingvals(typeof(am.spec), am.data)
    am.dist.coefs.=startingvals(typeof(am.dist), am.data)
    am.meanspec.coefs.=startingvals(typeof(am.meanspec), am.data)
    _fit!(am.spec.coefs, am.dist.coefs, am.meanspec.coefs, typeof(am.spec),
         typeof(am.dist), typeof(am.meanspec), am.data; algorithm=algorithm,
         autodiff=autodiff, kwargs...
         )
	am.fitted=true
    am
end

"""
    fit(am::ARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)

Fit the ARCHModel specified by `am` and return the result in a new instance of
ARCHModel. Keyword arguments are passed on to the optimizer.
"""
function fit(am::ARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)
    am2=deepcopy(am)
    fit!(am2; algorithm=algorithm, autodiff=autodiff, kwargs...)
    return am2
end

"""
    selectmodel(::Type{VS}, data; kwargs...) -> ARCHModel

Fit the volatility specification `VS` with varying lag lengths and return that which
minimizes the [BIC](https://en.wikipedia.org/wiki/Bayesian_information_criterion).

# Keyword arguments:
- `dist=StdNormal`: the error distribution.
- `meanspec=Intercept`: the mean specification.
- `maxlags=3`: maximum lag length to try in each parameter of `VS`.
- `criterion=bic`: function that takes an `ARCHModel` and returns the criterion to minimize.
- `show_trace=false`: print `criterion` to screen for each estimated model.
- `algorithm=BFGS(), autodiff=:forward, kwargs...`: passed on to the optimizer.

# Example
```jldoctest
julia> selectmodel(EGARCH, BG96)

EGARCH{1,1,2} model with Gaussian errors, T=1974.


Mean equation parameters:

        Estimate  Std.Error   z value Pr(>|z|)
μ    -0.00900018 0.00943948 -0.953461   0.3404

Volatility parameters:

       Estimate Std.Error   z value Pr(>|z|)
ω    -0.0544398 0.0592073 -0.919478   0.3578
γ₁   -0.0243368 0.0270414 -0.899985   0.3681
β₁     0.960301 0.0388183   24.7384   <1e-99
α₁     0.405788  0.067466    6.0147    <1e-8
α₂    -0.207357  0.114161  -1.81636   0.0693
```
"""
function selectmodel(::Type{VS}, data::Vector{T};
                     dist::Type{SD}=StdNormal{T}, meanspec::Type{MS}=Intercept{T},
                     maxlags=3, criterion=bic, show_trace=false, algorithm=BFGS(),
                     autodiff=:forward, kwargs...
                     ) where {VS<:VolatilitySpec, T<:AbstractFloat,
                              SD<:StandardizedDistribution, MS<:MeanSpec
                              }
    mylock=Threads.SpinLock()
    ndims = my_unwrap_unionall(VS)-1#e.g., two (p and q) for GARCH{p, q, T}
    res = Array{ARCHModel, ndims}(undef, ntuple(i->maxlags, ndims))
    Threads.@threads for ind in collect(CartesianIndices(size(res)))
        res[ind] = fit(VS{ind.I...}, data; dist=dist, meanspec=meanspec,
                       algorithm=algorithm, autodiff=autodiff, kwargs...)
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

"""
    fit(::Type{SD}, data; algorithm=BFGS(), kwargs...)

Fit a standardized distribution to the data, using the MLE. Keyword arguments
are passed on to the optimizer.
"""
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

#for rand to work
Base.eltype(::StandardizedDistribution{T}) where {T} = T

#count the number of type vars. there's probably a better way.
function my_unwrap_unionall(@nospecialize a)
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

function show(io::IO, spec::VolatilitySpec)
    println(io, modname(typeof(spec)), " specification.\n\n", CoefTable(spec.coefs, coefnames(typeof(spec)), ["Parameters:"]))
end
function show(io::IO, am::ARCHModel)
	if isfitted(am)
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
	   length(am.meanspec.coefs) > 0 && println(io, CoefTable(am.meanspec.coefs, coefnames(typeof(am.meanspec)), ["Mean equation parameters:"]))
	   println(io, CoefTable(am.spec.coefs, coefnames(typeof(am.spec)), ["Volatility parameters:   "]))
	   length(am.dist.coefs) > 0 && println(io, CoefTable(am.dist.coefs, coefnames(typeof(am.dist)), ["Distribution parameters: "]))
   end
end

#from here https://stackoverflow.com/questions/46671965/printing-variable-subscripts-in-julia
subscript(i::Integer) = i<0 ? error("$i is negative") : join('₀'+d for d in reverse(digits(i)))

function modname(::Type{VS}) where VS<:VolatilitySpec
    s = "$(VS)"
    s = s[1:findlast(isequal(','), s)-1] * '}'
end

include("meanspecs.jl")
include("standardizeddistributions.jl")
include("GARCH.jl")
include("EGARCH.jl")

#below is the fastest implementation that I can come up with, for reference.
#instead of keeping σ²ₜ in CircularBuffer, it keeps them in scalars and thus
#on the stack.
using Base.Cartesian: @nexprs
function fastfit(VS, data)
    obj = x-> fastmLL(VS, x, data, var(data))
    optimize(obj,
             startingvals(VS, data),
             BFGS(), autodiff=:forward, Optim.Options(x_tol=1e-4)
             )
end


function _fastmLL(VS::Type{GARCH{p,q, T1}}) where {p, q, T1}
    r=max(p, q)
    quote
        lower, upper = constraints(VS, T1)
        all(lower.<coefs.<upper) || return T2(-Inf)
        @inbounds begin
            @nexprs $r i -> h_i=h
            T = length(data)
            LL = zero(T2)
            @nexprs $r i -> (a_{$r+1-i} = data[i]; asq_{$r+1-i} = a_{$r+1-i}*a_{$r+1-i}; LL += log(abs(h_{$r+1-i}))+asq_{$r+1-i}/h_{$r+1-i})
            for t = $r+1:T
                h = coefs[1]
                @nexprs $p i -> (h += coefs[i+1]*h_i)
                @nexprs $q i -> (h += coefs[i+$(1+p)]*asq_i)
                @nexprs $(r-1) i-> (h_{$r+1-i}=h_{$r-i})
                @nexprs $(r-1) i-> (asq_{$r+1-i}=asq_{$r-i})
                h_1 = h
                a_1 = data[t]
                asq_1 = a_1*a_1
                LL += log(abs(h_1))+asq_1/h_1
            end
        end
        LL += T*T2(1.8378770664093453) #log2π
        LL *= .5
    end
end
@generated function fastmLL(::Type{VS}, coefs::AbstractVector{T2}, data::Vector{T1}, h) where {VS, T2, T1}
    return _fastmLL(VS{T1})
end
##example: GARCH{2, 2}
function fastmLL2(coef::AbstractVector{T2}, data, h) where {T2}
    h1 = h
    h2 = h
    T = length(data)
    FF = Float64[]
    LL = zero(T2)
    @inbounds a2 = data[1]
    asq2 = a2*a2
    LL += .5*log(abs(h2))-logkernel(StdNormal{Float64}, a2/sqrt(h2), FF)
    @inbounds a1 = data[2]
    asq1 = a1*a1
    LL += .5*log(abs(h1))-logkernel(StdNormal{Float64}, a1/sqrt(h1), FF)

    @inbounds for t = 3:T
        h = coef[1]+coef[2]*h1+coef[3]*h2+coef[4]*asq1+coef[5]*asq2
        h < 0 && return T2(Inf)
        h2 = h1
        asq2 = asq1
        h1 = h
        a1 = data[t]
        asq1 = a1*a1
        LL += .5*log(abs(h1))-logkernel(StdNormal{Float64}, a1/sqrt(h1), FF)
    end
    LL -= T*logconst(StdNormal, FF)
end

end#module
