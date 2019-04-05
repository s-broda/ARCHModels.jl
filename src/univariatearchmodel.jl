"""
    BG96
Data from [Bollerslev and Ghysels (JBES 1996)](https://doi.org/10.2307/1392425).
"""
const BG96 = readdlm(joinpath(dirname(pathof(ARCHModels)), "data", "bollerslev_ghysels.txt"), skipstart=1)[:, 1];

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
                                      am.meanspec, am.data,
                                      vcat(am.spec.coefs, am.dist.coefs,
                                           am.meanspec.coefs
                                           )
                                      )

dof(am::UnivariateARCHModel) = nparams(typeof(am.spec)) + nparams(typeof(am.dist)) + nparams(typeof(am.meanspec))
coef(am::UnivariateARCHModel)=vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs)
coefnames(am::UnivariateARCHModel) = vcat(coefnames(typeof(am.spec)),
                                coefnames(typeof(am.dist)),
                                coefnames(am.meanspec)
                                )


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
	@assert warmup>=0
	append!(data, zeros(T2, warmup))
    T = length(data)
	r1 = presample(typeof(spec))
	r2 = presample(meanspec)
	r = max(r1, r2)
	r = max(r, 1) # make sure this works for, e.g., ARCH{0}; CircularBuffer requires at least a length of 1
    ht = CircularBuffer{T2}(r)
    lht = CircularBuffer{T2}(r)
    zt = CircularBuffer{T2}(r)
	at = CircularBuffer{T2}(r)
    @inbounds begin
        h0 = uncond(typeof(spec), spec.coefs)
		m0 = uncond(meanspec)
        h0 > 0 || error("Model is nonstationary.")
        for t = 1:T
			if t>r2
				themean = mean(at, ht, lht, data, meanspec, meanspec.coefs, t)
			else
				themean = m0
			end
			if t>r1
                update!(ht, lht, zt, at, typeof(spec), meanspec,
                        data, spec.coefs, meanspec.coefs
                        )
            else
				push!(ht, h0)
                push!(lht, log(h0))
            end
			push!(zt, rand(dist))
			push!(at, sqrt(ht[end])*zt[end])
			data[t] = themean + at[end]
        end
    end
    deleteat!(data, 1:warmup)
end

@inline function splitcoefs(coefs, VS, SD, meanspec)
    ng = nparams(VS)
    nd = nparams(SD)
    nm = nparams(typeof(meanspec))
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
function volatilities(am::UnivariateARCHModel{T, VS, SD}) where {T, VS, SD}
	ht = Vector{T}(undef, 0)
	lht = Vector{T}(undef, 0)
	zt = Vector{T}(undef, 0)
	at = Vector{T}(undef, 0)
	loglik!(ht, lht, zt, at, VS, SD, am.meanspec, am.data, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
	return sqrt.(ht)
end

"""
    predict(am::UnivariateARCHModel, what=:volatility; level=0.01)
Form a 1-step ahead prediction from `am`. `what` controls which object is predicted.
The choices are `:volatility` (the default), `:variance`, `:return`, and `:VaR`. The VaR
level can be controlled with the keyword argument `level`.
"""
function predict(am::UnivariateARCHModel{T, VS, SD}, what=:volatility; level=0.01) where {T, VS, SD, MS}
	ht = volatilities(am).^2
	lht = log.(ht)
	zt = residuals(am)
	at = residuals(am, standardized=false)
	t = length(am.data) + 1

	if what == :return || what == :VaR
		themean = mean(at, ht, lht, am.data, am.meanspec, am.meanspec.coefs, t)
	end
	update!(ht, lht, zt, at, VS, am.meanspec, am.data, am.spec.coefs, am.meanspec.coefs)
	if what == :return
		return themean
	elseif what == :volatility
		return sqrt(ht[end])
	elseif what == :variance
		return ht[end]
	elseif what == :VaR
		return -themean - sqrt(ht[end]) * quantile(am.dist, level)
	else error("Prediction target $what unknown.")
	end
end

"""
    means(am::UnivariateARCHModel)
Return the conditional means of the model.
"""
function means(am::UnivariateARCHModel)
	return am.data-residuals(am; standardized=false)
end

"""
    residuals(am::UnivariateARCHModel; standardized=true)
Return the residuals of the model. Pass `standardized=false` for the non-devolatized residuals.
"""
function residuals(am::UnivariateARCHModel{T, VS, SD}; standardized=true) where {T, VS, SD}
		ht = Vector{T}(undef, 0)
		lht = Vector{T}(undef, 0)
		zt = Vector{T}(undef, 0)
		at = Vector{T}(undef, 0)
		loglik!(ht, lht, zt, at, VS, SD, am.meanspec, am.data, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
	return standardized ? zt : at
end

"""
    VaRs(am::UnivariateARCHModel, level=0.01)
Return the in-sample Value at Risk implied by `am`.
"""
function VaRs(am::UnivariateARCHModel, level=0.01)
    return -means(am) .- volatilities(am) .* quantile(am.dist, level)
end

#this works on CircularBuffers. The idea is that ht/lht/zt need to be allocated
#inside of this function, when the type that Optim it with is known (because
#it calls it with dual numbers for autodiff to work). It works with arrays, too,
#but grows them by length(data); hence it should be called with an empty one-
#dimensional array of the right type.
@inline function loglik!(ht::AbstractVector{T2}, lht::AbstractVector{T2},
                         zt::AbstractVector{T2}, at::AbstractVector{T2}, ::Type{VS}, ::Type{SD}, meanspec::MS,
                         data::Vector{T1}, coefs::AbstractVector{T2}
                         ) where {VS<:VolatilitySpec, SD<:StandardizedDistribution,
                                  MS<:MeanSpec, T1<:AbstractFloat, T2
                                  }
    garchcoefs, distcoefs, meancoefs = splitcoefs(coefs, VS, SD, meanspec)
    #the below 6 lines can be removed when using Fminbox
    lowergarch, uppergarch = constraints(VS, T1)
    lowerdist, upperdist = constraints(SD, T1)
    lowermean, uppermean = constraints(MS, T1)
    lower = vcat(lowergarch, lowerdist, lowermean)
    upper = vcat(uppergarch, upperdist, uppermean)
    all(lower.<coefs.<upper) || return T2(-Inf)
    T = length(data)
	r1 = presample(VS)
	r2 = presample(meanspec)
    r = max(r1, r2)
    T > r || error("Sample too small.")
	ki = kernelinvariants(SD, distcoefs)
    @inbounds begin
        h0 = var(data) # could be moved outside
		m0 = mean(data)
        #h0 = uncond(VS, garchcoefs)
        #h0 > 0 || return T2(NaN)
        LL = zero(T2)
        for t = 1:T
			if t>r2
				themean = mean(at, ht, lht, data, meanspec, meancoefs, t)
			else
				themean = m0
			end
			if t > r1
                update!(ht, lht, zt, at, VS, meanspec, data, garchcoefs, meancoefs)
            else
				push!(ht, h0)
                push!(lht, log(h0))
            end
            ht[end] < 0 && return T2(NaN)
			push!(at, data[t]-themean)
			push!(zt, at[end]/sqrt(ht[end]))
			LL += -lht[end]/2 + logkernel(SD, zt[end], distcoefs, ki...)

        end#for
    end#inbounds
    LL += T*logconst(SD, distcoefs)
end#function

function loglik(spec::Type{VS}, dist::Type{SD}, meanspec::MS,
                   data::Vector{<:AbstractFloat}, coefs::AbstractVector{T2}
                   ) where {VS<:VolatilitySpec, SD<:StandardizedDistribution,
                            MS<:MeanSpec, T2
                            }
    r = max(presample(VS), presample(meanspec))
	r = max(r, 1) # make sure this works for, e.g., ARCH{0}; CircularBuffer requires at least a length of 1
    ht = CircularBuffer{T2}(r)
    lht = CircularBuffer{T2}(r)
    zt = CircularBuffer{T2}(r)
	at = CircularBuffer{T2}(r)
    loglik!(ht, lht, zt, at, spec, dist, meanspec, data, coefs)

end

function logliks(spec, dist, meanspec, data, coefs::Vector{T}) where {T}
    garchcoefs, distcoefs, meancoefs = splitcoefs(coefs, spec, dist, meanspec)
    ht = T[]
    lht = T[]
    zt = T[]
	at = T[]
    loglik!(ht, lht, zt, at, spec, dist, meanspec, data, coefs)
    LLs = -lht./2 .+ logkernel.(dist, zt, Ref{Vector{T}}(distcoefs), kernelinvariants(dist, distcoefs)...) .+ logconst(dist, distcoefs)
end

function informationmatrix(am::UnivariateARCHModel; expected::Bool=true)
	expected && error("expected informationmatrix is not implemented for UnivariateARCHModel. Use expected=false.")
	g = x -> sum(logliks(typeof(am.spec), typeof(am.dist), am.meanspec, am.data, x))
	H = ForwardDiff.hessian(g, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
	J = -H/nobs(am)
end

function scores(am::UnivariateARCHModel)
	f = x -> logliks(typeof(am.spec), typeof(am.dist), am.meanspec, am.data, x)
	S = ForwardDiff.jacobian(f, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
end


function _fit!(garchcoefs::Vector{T}, distcoefs::Vector{T},
              meancoefs::Vector{T}, ::Type{VS}, ::Type{SD}, meanspec::MS,
              data::Vector{T}; algorithm=BFGS(), autodiff=:forward, kwargs...
              ) where {VS<:VolatilitySpec, SD<:StandardizedDistribution,
                       MS<:MeanSpec, T<:AbstractFloat
                       }
    obj = x -> -loglik(VS, SD, meanspec, data, x)
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
    nm = nparams(typeof(meanspec))
    garchcoefs .= coefs[1:ng]
    distcoefs .= coefs[ng+1:ng+ns]
    meancoefs .= coefs[ng+ns+1:ng+ns+nm]
	meanspec.coefs .= meancoefs
    return nothing
end

"""
    fit(VS::Type{<:VolatilitySpec}, data; dist=StdNormal, meanspec=Intercept,
        algorithm=BFGS(), autodiff=:forward, kwargs...)

Fit the ARCH model specified by `VS` to `data`. `data` can be a vector or a
GLM.LinearModel (or GLM.DataFrameRegressionModel).

# Keyword arguments:
- `dist=StdNormal`: the error distribution.
- `meanspec=Intercept`: the mean specification, either as a type or instance of that type.
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
function fit end

function fit(::Type{VS}, data::Vector{T}; dist::Type{SD}=StdNormal{T},
             meanspec::Union{MS, Type{MS}}=Intercept{T}(T[0]), algorithm=BFGS(),
             autodiff=:forward, kwargs...
             ) where {VS<:VolatilitySpec, SD<:StandardizedDistribution,
                      MS<:MeanSpec, T<:AbstractFloat
                      }
	#can't use dispatch for this b/c meanspec is a kwarg
	meanspec isa Type ? ms = meanspec(zeros(T, nparams(meanspec))) : ms = deepcopy(meanspec)
    coefs = startingvals(VS, data)
    distcoefs = startingvals(SD, data)
    meancoefs = startingvals(ms, data)
	_fit!(coefs, distcoefs, meancoefs, VS, SD, ms, data; algorithm=algorithm, autodiff=autodiff, kwargs...)
	return UnivariateARCHModel(VS(coefs), data; dist=SD(distcoefs), meanspec=ms, fitted=true)
end

"""
    fit!(am::UnivariateARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)

Fit the UnivariateARCHModel specified by `am`, modifying `am` in place. Keyword arguments
are passed on to the optimizer.
"""
function fit!(am::UnivariateARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)
    am.spec.coefs.=startingvals(typeof(am.spec), am.data)
    am.dist.coefs.=startingvals(typeof(am.dist), am.data)
    am.meanspec.coefs.=startingvals(am.meanspec, am.data)
	_fit!(am.spec.coefs, am.dist.coefs, am.meanspec.coefs, typeof(am.spec),
         typeof(am.dist), am.meanspec, am.data; algorithm=algorithm,
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
- `meanspec=Intercept`: the mean specification, either as a type or instance of that type.
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
                     dist::Type{SD}=StdNormal{T}, meanspec::Union{MS, Type{MS}}=Intercept{T},
                     maxlags=3, criterion=bic, show_trace=false, algorithm=BFGS(),
                     autodiff=:forward, kwargs...
                     ) where {VS<:VolatilitySpec, T<:AbstractFloat,
                              SD<:StandardizedDistribution, MS<:MeanSpec
                              }
	#threading sometimes segfaults in tests locally. possibly https://github.com/JuliaLang/julia/issues/29934
	mylock=Threads.SpinLock()
    ndims = max(my_unwrap_unionall(VS)-1, 0)#e.g., two (p and q) for GARCH{p, q, T}
	ndims2 = max(my_unwrap_unionall(MS)-1, 0)#e.g., two (p and q) for ARMA{p, q, T}
    res = Array{UnivariateARCHModel, ndims+ndims2}(undef, ntuple(i->maxlags, ndims+ndims2))
    Threads.@threads for ind in collect(CartesianIndices(size(res)))
		VSi = VS{ind.I[1:ndims]...}
		MSi = (ndims2==0 ? meanspec : meanspec{ind.I[ndims+1:end]...})
		res[ind] = fit(VSi, data; dist=dist, meanspec=MSi,
                       algorithm=algorithm, autodiff=autodiff, kwargs...)
        if show_trace
            lock(mylock)
            Core.print(modname(VSi))
			ndims2>0 && Core.print("-", modname(MSi))
			Core.println(" model has ",
                              uppercase(split("$criterion", ".")[end]), " ",
                              criterion(res[ind]), "."
                              )
            unlock(mylock)
        end
    end
    crits = criterion.(res)
    _, ind = findmin(crits)
    return res[ind]
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
	                               typeof(am.dist), am.meanspec
	                               )
	    seg, sed, sem = splitcoefs(se, typeof(am.spec),
	                               typeof(am.dist), am.meanspec
	                               )
	    zzg = ccg ./ seg
	    zzd = ccd ./ sed
	    zzm = ccm ./ sem
	    println(io, "\n", modname(typeof(am.spec)), " model with ",
	            distname(typeof(am.dist)), " errors, T=", nobs(am), ".\n\n")

	    length(sem) > 0 && println(io, "Mean equation parameters:", "\n\n",
	                               CoefTable(hcat(ccm, sem, zzm, 2.0 * normccdf.(abs.(zzm))),
	                                         ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
	                                         coefnames(am.meanspec), 4
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
	   length(am.meanspec.coefs) > 0 && println(io, CoefTable(am.meanspec.coefs, coefnames(am.meanspec), ["Mean equation parameters:"]))
	   println(io, CoefTable(am.spec.coefs, coefnames(typeof(am.spec)), ["Volatility parameters:   "]))
	   length(am.dist.coefs) > 0 && println(io, CoefTable(am.dist.coefs, coefnames(typeof(am.dist)), ["Distribution parameters: "]))
   end
end

function modname(::Type{S}) where S<:Union{VolatilitySpec, MeanSpec}
    s = "$(S)"
	lastcomma = findlast(isequal(','), s)
    lastcomma == nothing || (s = s[1:lastcomma-1] * '}')
	s
end
