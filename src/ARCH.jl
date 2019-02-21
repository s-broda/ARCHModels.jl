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
#the constructor for UnivariateARCHModel should make a copy of its args
#implement lrtest
#allow uninititalized constructors for VolatilitySpec, MeanSpec and StandardizedDistribution? If so, then be consistent with how they are defined
#  (change for meanspec and dist ), document, and test. Also, NaN is prob. safer than undef.
#constructors for meanspec, distributions should check length of coef vector
#logconst needs to return the correct type
#rename ARCH -> ARCHModels, _ARCH-> ARCH
"""
The ARCH package for Julia. For documentation, see https://s-broda.github.io/ARCH.jl/dev.
"""
module ARCH
using Reexport
@reexport using StatsBase
using StatsFuns: normcdf, normccdf, normlogpdf, norminvcdf, log2π, logtwo, RFunctions.tdistrand, RFunctions.tdistinvcdf, RFunctions.gammarand, RFunctions.gammainvcdf
using SpecialFunctions: beta, lgamma, gamma
using Optim
using ForwardDiff
using Distributions
using HypothesisTests
using Roots
using LinearAlgebra
using DataStructures: CircularBuffer
using DelimitedFiles
import Distributions: quantile
import Base: show, showerror, eltype
import Statistics: mean
import Random: rand
import HypothesisTests: HypothesisTest, testname, population_param_of_interest, default_tail, show_params, pvalue
import StatsBase: StatisticalModel, stderror, loglikelihood, nobs, fit, fit!, confint, aic,
                  bic, aicc, dof, coef, coefnames, coeftable, CoefTable,
				  informationmatrix, islinear, score, vcov, residuals, predict

export ARCHModel, UnivariateARCHModel, VolatilitySpec, StandardizedDistribution, Standardized, MeanSpec,
       simulate, simulate!, selectmodel, StdNormal, StdT, StdGED, Intercept,
       NoIntercept, BG96, volatilities, mean, quantile, VaRs, pvalue

"""
    BG96
Data from [Bollerslev and Ghysels (JBES 1996)](https://doi.org/10.2307/1392425).
"""
const BG96 = readdlm(joinpath(dirname(pathof(ARCH)), "data", "bollerslev_ghysels.txt"), skipstart=1)[:, 1];

"""
	ARCHModel <: StatisticalModel
"""
abstract type ARCHModel <: StatisticalModel end

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
    UnivariateARCHModel{T<:AbstractFloat,
              		    VS<:VolatilitySpec,
              			SD<:StandardizedDistribution{T},
              			MS<:MeanSpec{T}
              			} <: StatisticalModel
"""
mutable struct UnivariateARCHModel{T<:AbstractFloat,
                 				   VS<:VolatilitySpec,
                 		  	  	   SD<:StandardizedDistribution{T},
                 				   MS<:MeanSpec{T}
                 				   } <: ARCHModel
    spec::VS
    data::Vector{T}
    dist::SD
    meanspec::MS
	fitted::Bool
    function UnivariateARCHModel{T, VS, SD, MS}(spec, data, dist, meanspec, fitted) where {T, VS, SD, MS}
        new(spec, data, dist, meanspec, fitted)
    end
end

"""
    UnivariateARCHModel(spec::VolatilitySpec, data::Vector; dist=StdNormal(),
	          			meanspec=NoIntercept(), fitted=false
              			)

Create a UnivariateARCHModel.

# Example:
```jldoctest
julia> UnivariateARCHModel(GARCH{1, 1}([1., .9, .05]), randn(10))

TGARCH{0,1,1} model with Gaussian errors, T=10.


                             ω  β₁   α₁
Volatility parameters:     1.0 0.9 0.05
```
"""
function UnivariateARCHModel(spec::VS,
          		 			 data::Vector{T};
          					 dist::SD=StdNormal{T}(),
          				 	 meanspec::MS=NoIntercept{T}(),
		  			 		 fitted::Bool=false
          					 ) where {T<:AbstractFloat,
                    			 	  VS<:VolatilitySpec,
                   					  SD<:StandardizedDistribution,
                   					  MS<:MeanSpec
                   			 		  }
    UnivariateARCHModel{T, VS, SD, MS}(spec, data, dist, meanspec, fitted)
end

loglikelihood(am::UnivariateARCHModel) = loglik(typeof(am.spec), typeof(am.dist),
                                      typeof(am.meanspec), am.data,
                                      vcat(am.spec.coefs, am.dist.coefs,
                                           am.meanspec.coefs
                                           )
                                      )

nobs(am::UnivariateARCHModel) = length(am.data)
dof(am::UnivariateARCHModel) = nparams(typeof(am.spec)) + nparams(typeof(am.dist)) + nparams(typeof(am.meanspec))
coef(am::UnivariateARCHModel)=vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs)
coefnames(am::UnivariateARCHModel) = vcat(coefnames(typeof(am.spec)),
                                coefnames(typeof(am.dist)),
                                coefnames(typeof(am.meanspec))
                                )
islinear(am::UnivariateARCHModel) = false

function confint(am::UnivariateARCHModel, level::Real=0.95)
    hcat(coef(am), coef(am)) .+ stderror(am)*quantile(Normal(),(1. -level)/2.)*[1. -1.]
end

isfitted(am::UnivariateARCHModel) = am.fitted

"""
    simulate(am::UnivariateARCHModel; warmup=100)
	simulate(am::UnivariateARCHModel, nobs; warmup=100)
    simulate(spec::VolatilitySpec, nobs; warmup=100, dist=StdNormal(), meanspec=NoIntercept())
Simulate a UnivariateARCHModel.
"""
function simulate end

simulate(am::UnivariateARCHModel; warmup=100) = simulate(am, nobs(am); warmup=warmup)

function simulate(am::UnivariateARCHModel, nobs; warmup=100)
	am2 = deepcopy(am)
    simulate(am2.spec, nobs; warmup=warmup, dist=am2.dist, meanspec=am2.meanspec)
end

function simulate(spec::VolatilitySpec{T2}, nobs; warmup=100, dist::StandardizedDistribution{T2}=StdNormal{T2}(),
                  meanspec::MeanSpec{T2}=NoIntercept{T2}()
                  ) where {T2<:AbstractFloat}
    data = zeros(T2, nobs)
    _simulate!(data,  spec; warmup=warmup, dist=dist, meanspec=meanspec)
    UnivariateARCHModel(spec, data; dist=dist, meanspec=meanspec, fitted=false)
end

"""
    simulate!(am::UnivariateARCHModel; warmup=100)
Simulate a UnivariateARCHModel, modifying `am` in place.
"""
function simulate!(am::UnivariateARCHModel; warmup=100)
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
    volatilities(am::UnivariateARCHModel)
Return the conditional volatilities.
"""
function volatilities(am::UnivariateARCHModel{T, VS, SD, MS}) where {T, VS, SD, MS}
	ht = Vector{T}(undef, 0)
	lht = Vector{T}(undef, 0)
	zt = Vector{T}(undef, 0)
	loglik!(ht, lht, zt, VS, SD, MS, am.data, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
	return sqrt.(ht)
end

"""
    predict(am::UnivariateARCHModel, what=:volatility; level=0.01)
Form a 1-step ahead prediction from `am`. `what` controls which object is predicted.
The choices are `:volatility` (the default), `:variance`, `:return`, and `:VaR`. The VaR
level can be controlled with the keyword argument `level`.
"""
function predict(am::UnivariateARCHModel{T, VS, SD, MS}, what=:volatility; level=0.01) where {T, VS, SD, MS}
	ht = volatilities(am).^2
	lht = log.(ht)
	zt = residuals(am)
	t = length(am.data)
	update!(ht, lht, zt, VS, MS, am.data, am.spec.coefs, am.meanspec.coefs, t)
	#this (and a loop) is what we'd need for n-step. but this will only work vor the variance, and only for GARCH:
	#push!(zt, zero(T))
	#push!(am.data, mean(am.meanspec))
	if what == :return
		return mean(am.meanspec)
	elseif what == :volatility
		return sqrt(ht[end])
	elseif what == :variance
		return ht[end]
	elseif what == :VaR
		return -mean(am.meanspec) - sqrt(ht[end]) * quantile(am.dist, level)
	else error("Prediction target $what unknown.")
	end
end

"""
    residuals(am::UnivariateARCHModel; standardized=true)
Return the residuals of the model. Pass `standardized=false` for the non-devolatized residuals.
"""
function residuals(am::UnivariateARCHModel{T, VS, SD, MS}; standardized=true) where {T, VS, SD, MS}
	if standardized
		ht = Vector{T}(undef, 0)
		lht = Vector{T}(undef, 0)
		zt = Vector{T}(undef, 0)
		loglik!(ht, lht, zt, VS, SD, MS, am.data, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
		return zt
	else
		return am.data.-mean(am.meanspec)
	end
end

"""
    VaRs(am::UnivariateARCHModel, level=0.01)
Return the in-sample Value at Risk implied by `am`.
"""
function VaRs(am::UnivariateARCHModel, level=0.01)
    return -mean(am.meanspec) .- volatilities(am) .* quantile(am.dist, level)
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
	ki = kernelinvariants(SD, distcoefs)
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
            LL += -lht[end]/2 + logkernel(SD, zt[end], distcoefs, ki...)
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
    LLs = -lht./2 .+ logkernel.(dist, zt, Ref{Vector{T}}(distcoefs), kernelinvariants(dist, distcoefs)...) .+ logconst(dist, distcoefs)
end

function informationmatrix(am::UnivariateARCHModel; expected::Bool=true)
	expected && error("expected informationmatrix is not implemented for UnivariateARCHModel. Use expected=false.")
	g = x -> sum(ARCH.logliks(typeof(am.spec), typeof(am.dist), typeof(am.meanspec), am.data, x))
	H = ForwardDiff.hessian(g, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
	J = -H/nobs(am)
end

function scores(am::UnivariateARCHModel)
	f = x -> ARCH.logliks(typeof(am.spec), typeof(am.dist), typeof(am.meanspec), am.data, x)
	S = ForwardDiff.jacobian(f, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
end

score(am::UnivariateARCHModel) = sum(scores(am), dims=1)


function vcov(am::UnivariateARCHModel)
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

stderror(am::UnivariateARCHModel) = sqrt.(abs.(diag(vcov(am))))

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
julia> fit(EGARCH{1, 1, 1}, BG96; meanspec=NoIntercept, dist=StdT)

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
	return UnivariateARCHModel(VS(coefs), data; dist=SD(distcoefs), meanspec=MS(meancoefs), fitted=true)
end

"""
    fit!(am::UnivariateARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)

Fit the UnivariateARCHModel specified by `am`, modifying `am` in place. Keyword arguments
are passed on to the optimizer.
"""
function fit!(am::UnivariateARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)
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
    fit(am::UnivariateARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)

Fit the UnivariateARCHModel specified by `am` and return the result in a new instance of
UnivariateARCHModel. Keyword arguments are passed on to the optimizer.
"""
function fit(am::UnivariateARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)
    am2=deepcopy(am)
    fit!(am2; algorithm=algorithm, autodiff=autodiff, kwargs...)
    return am2
end

"""
    selectmodel(::Type{VS}, data; kwargs...) -> UnivariateARCHModel

Fit the volatility specification `VS` with varying lag lengths and return that which
minimizes the [BIC](https://en.wikipedia.org/wiki/Bayesian_information_criterion).

# Keyword arguments:
- `dist=StdNormal`: the error distribution.
- `meanspec=Intercept`: the mean specification.
- `maxlags=3`: maximum lag length to try in each parameter of `VS`.
- `criterion=bic`: function that takes a `UnivariateARCHModel` and returns the criterion to minimize.
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
	#threading sometimes segfaults in tests locally. possibly https://github.com/JuliaLang/julia/issues/29934
    mylock=Threads.SpinLock()
    ndims = my_unwrap_unionall(VS)-1#e.g., two (p and q) for GARCH{p, q, T}
    res = Array{UnivariateARCHModel, ndims}(undef, ntuple(i->maxlags, ndims))
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

#count the number of type vars. there's probably a better way.
function my_unwrap_unionall(@nospecialize a)
    count = 0
    while isa(a, UnionAll)
        a = a.body
        count += 1
    end
    return count
end

function coeftable(am::UnivariateARCHModel)
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
function show(io::IO, am::UnivariateARCHModel)
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
include("EGARCH.jl")
include("TGARCH.jl")
include("tests.jl")

## some speed experiments
# using Base.Cartesian: @nexprs
#
# @generated function loglik2(::Type{VS}, ::Type{SD}, ::Type{MS},
# 							data::Vector{T1}, coefs::AbstractVector{T2}
# 							) where {VS<:VolatilitySpec, SD<:StandardizedDistribution,
# 									 MS<:MeanSpec, T1<:AbstractFloat, T2
# 									 }
# 	r = presample(VS)
# 	lowergarch, uppergarch = constraints(VS, T1)
# 	lowerdist, upperdist = constraints(SD, T1)
# 	lowermean, uppermean = constraints(MS, T1)
# 	lower = vcat(lowergarch, lowerdist, lowermean)
# 	upper = vcat(uppergarch, upperdist, uppermean)
# 	quote
# 		garchcoefs, distcoefs, meancoefs = splitcoefs(coefs, VS, SD, MS)
# 		all($lower.<coefs.<$upper) || return T2(-Inf)
# 		T = length(data)
# 		T > $r || error("Sample too small.")
# 		ki = kernelinvariants(SD, distcoefs)
# 		@inbounds begin
# 			h0 = var(data)
# 			h0 < 0 && return T2(NaN)
# 			lh0 = log(h0)
# 			LL = zero(T2)
# 			m = mean(MS, meancoefs)
# 			@nexprs $r i -> h_i=h0
# 			@nexprs $r i -> lh_i=log(h0)
# 			@nexprs $r i -> a_{$r+1-i} = data[i]-m
# 			@nexprs $r i -> z_{$r+1-i} = a_{$r+1-i}/sqrt(h_{$r+1-i})
# 			@nexprs $r i -> LL += -lh0/2+logkernel(SD, z_{$r+1-i}, distcoefs, ki...)
# 			for t = $r+1:T
# 				$(update(VS))
# 				@nexprs $(r-1) i-> (h_{$r+1-i}=h_{$r-i})
# 				@nexprs $(r-1) i-> (lh_{$r+1-i}=lh_{$r-i})
# 				@nexprs $(r-1) i-> (a_{$r+1-i}=a_{$r-i})
# 				@nexprs $(r-1) i-> (z_{$r+1-i}=z_{$r-i})
# 				h_1 = h
# 				lh_1 = lh
# 				a_1 = data[t]-m
# 				z_1 = a_1/sqrt(h_1)
# 				LL += -lh/2 + logkernel(SD, z_1, distcoefs, ki...)
# 			end
# 		end#inbounds
# 		LL += T*logconst(SD, distcoefs)
# 	end #quote
# end#function
#
#
# function update(::Type{<:GARCH{p, q, T2} where T2} ) where {p, q}
# 	quote
# 		h = coefs[1]
# 		@nexprs $p i -> (h += coefs[i+1]*h_i)
# 		@nexprs $q i -> (h += coefs[i+$(1+p)]*a_i^2)
# 		h < 0 && return T2(NaN)
# 		lh = log(h)
# 	end
# end
#
# function update(::Type{<:EGARCH{o, p, q, T2} where T2} ) where {o, p, q}
# 	quote
# 		lh = coefs[1]
# 		@nexprs $o i -> (lh += coefs[i+1]*z_i)
# 		@nexprs $p i -> (lh += coefs[i+$(1+o)]*lh_i)
# 		@nexprs $q i -> (lh += coefs[i+$(1+p+o)]*(abs(z_i) - sqrt2invpi))
# 		h = exp(lh)
# 	end
# end
# #below is the fastest implementation that I can come up with, for reference.
# #instead of keeping σ²ₜ in CircularBuffer, it keeps them in scalars and thus
# #on the stack.
#
# function fastfit(VS, data)
#     obj = x-> fastmLL(VS, x, data, var(data))
#     optimize(obj,
#              startingvals(VS, data),
#              BFGS(), autodiff=:forward
#              )
# end
#
#
# function _fastmLL(VS::Type{GARCH{p,q, T1}}) where {p, q, T1}
#     r=max(p, q)
#     quote
#         lower, upper = constraints(VS, T1)
#         all(lower.<coefs.<upper) || return T2(-Inf)
#         @inbounds begin
#             @nexprs $r i -> h_i=h
#             T = length(data)
#             LL = zero(T2)
#             @nexprs $r i -> (a_{$r+1-i} = data[i]; asq_{$r+1-i} = a_{$r+1-i}*a_{$r+1-i}; LL += log(abs(h_{$r+1-i}))+asq_{$r+1-i}/h_{$r+1-i})
#             for t = $r+1:T
#                 h = coefs[1]
#                 @nexprs $p i -> (h += coefs[i+1]*h_i)
#                 @nexprs $q i -> (h += coefs[i+$(1+p)]*asq_i)
#                 @nexprs $(r-1) i-> (h_{$r+1-i}=h_{$r-i})
#                 @nexprs $(r-1) i-> (asq_{$r+1-i}=asq_{$r-i})
#                 h_1 = h
#                 a_1 = data[t]
#                 asq_1 = a_1*a_1
#                 LL += log(abs(h_1))+asq_1/h_1
#             end
#         end
#         LL += T*T2(1.8378770664093453) #log2π
#         LL *= .5
#     end
# end
# @generated function fastmLL(::Type{VS}, coefs::AbstractVector{T2}, data::Vector{T1}, h) where {VS, T2, T1}
#     return _fastmLL(VS{T1})
# end
# ##example: GARCH{2, 2}
# function fastmLL2(coef::AbstractVector{T2}, data, h) where {T2}
#     h1 = h
#     h2 = h
#     T = length(data)
#     FF = Float64[]
#     LL = zero(T2)
#     @inbounds a2 = data[1]
#     asq2 = a2*a2
#     LL += .5*log(abs(h2))-logkernel(StdNormal{Float64}, a2/sqrt(h2), FF)
#     @inbounds a1 = data[2]
#     asq1 = a1*a1
#     LL += .5*log(abs(h1))-logkernel(StdNormal{Float64}, a1/sqrt(h1), FF)
#
#     @inbounds for t = 3:T
#         h = coef[1]+coef[2]*h1+coef[3]*h2+coef[4]*asq1+coef[5]*asq2
#         h < 0 && return T2(Inf)
#         h2 = h1
#         asq2 = asq1
#         h1 = h
#         a1 = data[t]
#         asq1 = a1*a1
#         LL += .5*log(abs(h1))-logkernel(StdNormal{Float64}, a1/sqrt(h1), FF)
#     end
#     LL -= T*logconst(StdNormal, FF)
# end

end#module
