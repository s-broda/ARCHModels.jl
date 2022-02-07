"""
    BG96
Data from [Bollerslev and Ghysels (JBES 1996)](https://doi.org/10.2307/1392425).
"""
const BG96 = readdlm(joinpath(dirname(pathof(ARCHModels)), "data", "bollerslev_ghysels.txt"), skipstart=1)[:, 1];

"""
    UnivariateVolatilitySpec{T} <: VolatilitySpec{T} end

Abstract supertype that univariate volatility specifications inherit from.
"""
abstract type UnivariateVolatilitySpec{T} <: VolatilitySpec{T} end

"""
    StandardizedDistribution{T} <: Distributions.Distribution{Univariate, Continuous}

Abstract supertype that standardized distributions inherit from.
"""
abstract type StandardizedDistribution{T} <: Distribution{Univariate, Continuous} end


"""
    UnivariateARCHModel{T<:AbstractFloat,
              		    VS<:UnivariateVolatilitySpec,
              			SD<:StandardizedDistribution{T},
              			MS<:MeanSpec{T}
              			} <: ARCHModel
"""
mutable struct UnivariateARCHModel{T<:AbstractFloat,
                 				   VS<:UnivariateVolatilitySpec,
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

mutable struct UnivariateSubsetARCHModel{T<:AbstractFloat,
                 				   VS<:UnivariateVolatilitySpec,
                 		  	  	   SD<:StandardizedDistribution{T},
                 				   MS<:MeanSpec{T},
								   N
                 				   } <: ARCHModel
	spec::VS
    data::Vector{T}
    dist::SD
    meanspec::MS
	fitted::Bool
	subset::NTuple{N, Int}
    function UnivariateSubsetARCHModel{T, VS, SD, MS, N}(spec, data, dist, meanspec, fitted, subset) where {T, VS, SD, MS, N}
        new(spec, data, dist, meanspec, fitted, subset)
    end
end

"""
    UnivariateARCHModel(spec::UnivariateVolatilitySpec, data::Vector; dist=StdNormal(),
	          			meanspec=NoIntercept(), fitted=false
              			)

Create a UnivariateARCHModel.

# Example:
```jldoctest
julia> UnivariateARCHModel(GARCH{1, 1}([1., .9, .05]), randn(10))

TGARCH{0,1,1} model with Gaussian errors, T=10.


─────────────────────────────────────────
                             ω   β₁    α₁
─────────────────────────────────────────
Volatility parameters:     1.0  0.9  0.05
─────────────────────────────────────────
```
"""
function UnivariateARCHModel(spec::VS,
          		 			 data::Vector{T};
          					 dist::SD=StdNormal{T}(),
          				 	 meanspec::MS=NoIntercept{T}(),
		  			 		 fitted::Bool=false
          					 ) where {T<:AbstractFloat,
                    			 	  VS<:UnivariateVolatilitySpec,
                   					  SD<:StandardizedDistribution,
                   					  MS<:MeanSpec
                   			 		  }
    UnivariateARCHModel{T, VS, SD, MS}(spec, data, dist, meanspec, fitted)
end

function UnivariateSubsetARCHModel(spec::VS,
          		 			 data::Vector{T};
          					 dist::SD=StdNormal{T}(),
          				 	 meanspec::MS=NoIntercept{T}(),
		  			 		 fitted::Bool=false,
							 subset::NTuple{N, Int}
							 ) where {T<:AbstractFloat,
                    			 	  VS<:UnivariateVolatilitySpec,
                   					  SD<:StandardizedDistribution,
                   					  MS<:MeanSpec,
									  N
                   			 		  }
    UnivariateSubsetARCHModel{T, VS, SD, MS, N}(spec, data, dist, meanspec, fitted, subset)
end
loglikelihood(am::UnivariateARCHModel) = loglik(typeof(am.spec), typeof(am.dist),
                                      am.meanspec, am.data,
                                      vcat(am.spec.coefs, am.dist.coefs,
                                           am.meanspec.coefs
                                           )
                                      )

loglikelihood(am::UnivariateSubsetARCHModel) = loglik(typeof(am.spec), typeof(am.dist),
                                      am.meanspec, am.data,
                                      vcat(am.spec.coefs, am.dist.coefs,
                                           am.meanspec.coefs
                                           ),
									  subsetmask(typeof(am.spec), am.subset)
									  )


dof(am::UnivariateARCHModel) = nparams(typeof(am.spec)) + nparams(typeof(am.dist)) + nparams(typeof(am.meanspec))
dof(am::UnivariateSubsetARCHModel) = nparams(typeof(am.spec), am.subset) + nparams(typeof(am.dist)) + nparams(typeof(am.meanspec))
coef(am::UnivariateARCHModel)=vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs)
coefnames(am::UnivariateARCHModel) = vcat(coefnames(typeof(am.spec)),
                                coefnames(typeof(am.dist)),
                                coefnames(am.meanspec)
                                )


# documented in general
function simulate(spec::UnivariateVolatilitySpec{T2}, nobs; warmup=100, dist::StandardizedDistribution{T2}=StdNormal{T2}(),
                  meanspec::MeanSpec{T2}=NoIntercept{T2}()
                  ) where {T2<:AbstractFloat}
    data = zeros(T2, nobs)
    _simulate!(data,  spec; warmup=warmup, dist=dist, meanspec=meanspec)
    UnivariateARCHModel(spec, data; dist=dist, meanspec=meanspec, fitted=false)
end

function _simulate!(data::Vector{T2}, spec::UnivariateVolatilitySpec{T2};
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
                update!(ht, lht, zt, at, typeof(spec), spec.coefs)
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
    predict(am::UnivariateARCHModel, what=:volatility, horizon=1; level=0.01)
Form a `horizon`-step ahead prediction from `am`. `what` controls which object is predicted.
The choices are `:volatility` (the default), `:variance`, `:return`, and `:VaR`. The VaR
level can be controlled with the keyword argument `level`.

Not all prediction targets / volatility specifications support multi-step predictions.
"""
function predict(am::UnivariateARCHModel{T, VS, SD}, what=:volatility, horizon=1; level=0.01) where {T, VS, SD, MS}
	ht = volatilities(am).^2
	lht = log.(ht)
	zt = residuals(am)
	at = residuals(am, standardized=false)
	themean = T(0)
	if horizon > 1
		if what == :VaR
			error("Predicting VaR more than one period ahead is not implemented. Consider predicting one period ahead and scaling by `sqrt(horizon)`.")
		elseif what == :volatility
			error("Predicting volatility more than one period ahead is not implemented.")
		elseif what == :variance && !(VS <: TGARCH)
			error("Predicting variance more than one period ahead is not implemented for $(modname(VS)).")
		end
	end
    data = copy(am.data)
	for current_horizon = (1 : horizon)
		t = length(data) + current_horizon
		if what == :return || what == :VaR
			themean = mean(at, ht, lht, am.data, am.meanspec, am.meanspec.coefs, t)
		end
		update!(ht, lht, zt, at, VS, am.spec.coefs, current_horizon)
		push!(zt, 0.)
		push!(at, 0.)
        push!(data, themean)
	end
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
                         zt::AbstractVector{T2}, at::AbstractVector{T2}, vs::Type{VS}, ::Type{SD}, meanspec::MS,
                         data::Vector{T1}, coefs::AbstractVector{T3}, subsetmask=trues(nparams(vs))
                         ) where {VS<:UnivariateVolatilitySpec, SD<:StandardizedDistribution,
                                  MS<:MeanSpec, T1<:AbstractFloat, T2, T3
                                  }
    garchcoefs, distcoefs, meancoefs = splitcoefs(coefs, VS, SD, meanspec)
	lowergarch, uppergarch = constraints(VS, T1)
	lowerdist, upperdist = constraints(SD, T1)
    lowermean, uppermean = constraints(MS, T1)
    all(lowerdist.<distcoefs.<upperdist) && all(lowermean.<meancoefs.<uppermean) && all(lowergarch[subsetmask].<garchcoefs[subsetmask].<uppergarch[subsetmask]) || return T2(-Inf)
	garchcoefs .*= subsetmask
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
                update!(ht, lht, zt, at, VS, garchcoefs)
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
                   data::Vector{<:AbstractFloat}, coefs::AbstractVector{T2}, subsetmask=trues(nparams(spec))
                   ) where {VS<:UnivariateVolatilitySpec, SD<:StandardizedDistribution,
                            MS<:MeanSpec, T2
                            }
    r = max(presample(VS), presample(meanspec))
	r = max(r, 1) # make sure this works for, e.g., ARCH{0}; CircularBuffer requires at least a length of 1
    ht = CircularBuffer{T2}(r)
    lht = CircularBuffer{T2}(r)
    zt = CircularBuffer{T2}(r)
	at = CircularBuffer{T2}(r)
    loglik!(ht, lht, zt, at, spec, dist, meanspec, data, coefs, subsetmask)

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
	H = FiniteDiff.finite_difference_hessian(g, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
	J = -H/nobs(am)
end

function scores(am::UnivariateARCHModel)
	f = x -> logliks(typeof(am.spec), typeof(am.dist), am.meanspec, am.data, x)
	S = ForwardDiff.jacobian(f, vcat(am.spec.coefs, am.dist.coefs, am.meanspec.coefs))
end


function _fit!(garchcoefs::Vector{T}, distcoefs::Vector{T},
              meancoefs::Vector{T}, ::Type{VS}, ::Type{SD}, meanspec::MS,
              data::Vector{T}; algorithm=BFGS(), autodiff=:forward, kwargs...
              ) where {VS<:UnivariateVolatilitySpec, SD<:StandardizedDistribution,
                       MS<:MeanSpec, T<:AbstractFloat
                       }
    obj = x -> -loglik(VS, SD, meanspec, data, x)
    coefs = vcat(garchcoefs, distcoefs, meancoefs)
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
    fit(VS::Type{<:UnivariateVolatilitySpec}, data; dist=StdNormal, meanspec=Intercept,
        algorithm=BFGS(), autodiff=:forward, kwargs...)

Fit the ARCH model specified by `VS` to `data`. `data` can be a vector or a
GLM.LinearModel (or GLM.TableRegressionModel).

# Keyword arguments:
- `dist=StdNormal`: the error distribution.
- `meanspec=Intercept`: the mean specification, either as a type or instance of that type.
- `algorithm=BFGS(), autodiff=:forward, kwargs...`: passed on to the optimizer.

# Example: EGARCH{1, 1, 1} model without intercept, Student's t errors.
```jldoctest
julia> fit(EGARCH{1, 1, 1}, BG96; meanspec=NoIntercept, dist=StdT)

EGARCH{1,1,1} model with Student's t errors, T=1974.


Volatility parameters:
─────────────────────────────────────────────
      Estimate  Std.Error   z value  Pr(>|z|)
─────────────────────────────────────────────
ω   -0.0162014  0.0186792  -0.86735    0.3858
γ₁  -0.0378454  0.0180239  -2.09974    0.0358
β₁   0.977687   0.0125567  77.862      <1e-99
α₁   0.255804   0.0625445   4.08995    <1e-4
─────────────────────────────────────────────

Distribution parameters:
─────────────────────────────────────────
   Estimate  Std.Error  z value  Pr(>|z|)
─────────────────────────────────────────
ν   4.12423    0.40059  10.2954    <1e-24
─────────────────────────────────────────
```
"""
function fit(::Type{VS}, data::Vector{T}; dist::Type{SD}=StdNormal{T},
             meanspec::Union{MS, Type{MS}}=Intercept{T}(T[0]), algorithm=BFGS(),
             autodiff=:forward, kwargs...
             ) where {VS<:UnivariateVolatilitySpec, SD<:StandardizedDistribution,
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

function fitsubset(::Type{VS}, data::Vector{T}, maxlags::Int, subset::Tuple; dist::Type{SD}=StdNormal{T},
             meanspec::Union{MS, Type{MS}}=Intercept{T}(T[0]), algorithm=BFGS(),
             autodiff=:forward, kwargs...
             ) where {VS<:UnivariateVolatilitySpec, SD<:StandardizedDistribution,
                      MS<:MeanSpec, T<:AbstractFloat
                      }
	#can't use dispatch for this b/c meanspec is a kwarg
	meanspec isa Type ? ms = meanspec(zeros(T, nparams(meanspec))) : ms = deepcopy(meanspec)
	VS_large = VS{ntuple(i->maxlags, length(subset))...}
	ng = nparams(VS_large)
	ns = nparams(SD)
	nm = nparams(typeof(ms))
	mask = subsetmask(VS_large, subset)
	garchcoefs = startingvals(VS_large, data, subset)
	distcoefs = startingvals(SD, data)
    meancoefs = startingvals(ms, data)

	obj = x -> -loglik(VS_large, SD, ms, data, x, mask)
    coefs = vcat(garchcoefs, distcoefs, meancoefs)
    res = optimize(obj, coefs, algorithm; autodiff=autodiff, kwargs...)
    coefs .= Optim.minimizer(res)
    garchcoefs .= coefs[1:ng]
	distcoefs .= coefs[ng+1:ng+ns]
    meancoefs .= coefs[ng+ns+1:ng+ns+nm]
	ms.coefs .= meancoefs
    return UnivariateSubsetARCHModel(VS_large(garchcoefs), data; dist=SD(distcoefs), meanspec=ms, fitted=true, subset=subset)
end

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

function fit(am::UnivariateARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)
    am2=deepcopy(am)
    fit!(am2; algorithm=algorithm, autodiff=autodiff, kwargs...)
    return am2
end

function fit(vs::Type{VS}, lm::TableRegressionModel{<:LinearModel}; kwargs...) where VS<:UnivariateVolatilitySpec
	fit(vs, response(lm.model); meanspec=Regression(modelmatrix(lm.model); coefnames=coefnames(lm)), kwargs...)
end

function fit(vs::Type{VS}, lm::LinearModel; kwargs...) where VS<:UnivariateVolatilitySpec
	fit(vs, response(lm); meanspec=Regression(modelmatrix(lm)), kwargs...)
end

"""
    selectmodel(::Type{VS}, data; kwargs...) -> UnivariateARCHModel

Fit the volatility specification `VS` with varying lag lengths and return that which
minimizes the [BIC](https://en.wikipedia.org/wiki/Bayesian_information_criterion).

# Keyword arguments:
- `dist=StdNormal`: the error distribution.
- `meanspec=Intercept`: the mean specification, either as a type or instance of that type.
- `minlags=1`: minimum lag length to try in each parameter of `VS`.
- `maxlags=3`: maximum lag length to try in each parameter of `VS`.
- `criterion=bic`: function that takes a `UnivariateARCHModel` and returns the criterion to minimize.
- `show_trace=false`: print `criterion` to screen for each estimated model.
- `algorithm=BFGS(), autodiff=:forward, kwargs...`: passed on to the optimizer.

# Example
```jldoctest
julia> selectmodel(EGARCH, BG96)

EGARCH{1,1,2} model with Gaussian errors, T=1974.

Mean equation parameters:
───────────────────────────────────────────────
      Estimate   Std.Error    z value  Pr(>|z|)
───────────────────────────────────────────────
μ  -0.00900018  0.00943934  -0.953475    0.3403
───────────────────────────────────────────────

Volatility parameters:
──────────────────────────────────────────────
      Estimate  Std.Error    z value  Pr(>|z|)
──────────────────────────────────────────────
ω   -0.0544398  0.0591898  -0.919751    0.3577
γ₁  -0.0243368  0.0270382  -0.900092    0.3681
β₁   0.960301   0.038806   24.7462      <1e-99
α₁   0.405788   0.0674641   6.01487     <1e-8
α₂  -0.207357   0.114132   -1.81682     0.0692
──────────────────────────────────────────────
```
"""
function selectmodel(::Type{VS}, data::Vector{T};
                     dist::Type{SD}=StdNormal{T}, meanspec::Union{MS, Type{MS}}=Intercept{T},
                     maxlags::Integer=3, minlags::Integer=1, criterion=bic, show_trace=false, algorithm=BFGS(),
                     autodiff=:forward, kwargs...
                     ) where {VS<:UnivariateVolatilitySpec, T<:AbstractFloat,
                              SD<:StandardizedDistribution, MS<:MeanSpec
                              }
	@assert maxlags >= minlags >= 0

	#threading sometimes segfaults in tests locally. possibly https://github.com/JuliaLang/julia/issues/29934
	mylock=Threads.ReentrantLock()
    ndims = max(my_unwrap_unionall(VS)-1, 0) # e.g., two (p and q) for GARCH{p, q, T}
	ndims2 = max(my_unwrap_unionall(MS)-1, 0 )# e.g., two (p and q) for ARMA{p, q, T}
    res = Array{UnivariateSubsetARCHModel, ndims+ndims2}(undef, ntuple(i->maxlags - minlags + 1, ndims+ndims2))
    Threads.@threads for ind in collect(CartesianIndices(size(res)))
		tup = (ind.I[1:ndims] .+ minlags .-1)
		MSi = (ndims2==0 ? deepcopy(meanspec) : meanspec{ind.I[ndims+1:end] .+ minlags .- 1...})
		res[ind] = fitsubset(VS, data, maxlags, tup; dist=dist, meanspec=MSi,
                       algorithm=algorithm, autodiff=autodiff, kwargs...)
        if show_trace
            lock(mylock)
			VSi = VS{tup...}
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
	return fit(VS{res[ind].subset...}, data; dist=dist, meanspec=res[ind].meanspec, algorithm=algorithm, autodiff=autodiff, kwargs...)
end

function coeftable(am::UnivariateARCHModel)
    cc = coef(am)
    se = stderror(am)
    zz = cc ./ se
    CoefTable(hcat(cc, se, zz, 2.0 * normccdf.(abs.(zz))),
              ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
              coefnames(am), 4)
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
	            distname(typeof(am.dist)), " errors, T=", nobs(am), ".\n")

	    length(sem) > 0 && println(io, "Mean equation parameters:", "\n",
	                               CoefTable(hcat(ccm, sem, zzm, 2.0 * normccdf.(abs.(zzm))),
	                                         ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
	                                         coefnames(am.meanspec), 4
	                                         )
	                              )
	    println(io, "\nVolatility parameters:", "\n",
	            CoefTable(hcat(ccg, seg, zzg, 2.0 * normccdf.(abs.(zzg))),
	                      ["Estimate", "Std.Error", "z value", "Pr(>|z|)"],
	                      coefnames(typeof(am.spec)), 4
	                      )
	            )
	    length(sed) > 0 && println(io, "\nDistribution parameters:", "\n",
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

function modname(::Type{S}) where S<:Union{UnivariateVolatilitySpec, MeanSpec}
    s = "$(S)"
	lastcomma = findlast(isequal(','), s)
    lastcomma == nothing || (s = s[1:lastcomma-1] * '}')
	s
end
