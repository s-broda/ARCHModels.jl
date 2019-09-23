# can consolidate the remaining simulate method if we have default meanspecs for univariate/multivariate
# proper multivariate meanspec, include return prediction in predict
# implement correlations, covariances, residuals in terms of update!, and move them from DCC to multivariate
"""
	DOW29
Stock returns, in procent, from 03/19/2008 through 04/11/2019, for tickers
AAPL, IBM, XOM, KO, MSFT, INTC, MRK, PG, VZ, WBA, V, JNJ, PFE, CSCO,
TRV, WMT, MMM, UTX, UNH, NKE, HD, BA, AXP, MCD, CAT, GS, JPM, CVX, DIS.
"""
const DOW29 = readdlm(joinpath(dirname(pathof(ARCHModels)), "data", "dow29.csv"), ',')

"""
    MultivariateStandardizedDistribution{T, d} <: Distribution{Multivariate, Continuous}

Abstract supertype that multivariate standardized distributions inherit from.
"""
abstract type MultivariateStandardizedDistribution{T, d} <: Distribution{Multivariate, Continuous} end

"""
    MultivariateVolatilitySpec{T, d} <: VolatilitySpec{T}

Abstract supertype that multivariate volatility specifications inherit from.
"""
abstract type MultivariateVolatilitySpec{T, d} <: VolatilitySpec{T} end

"""
	MultivariateARCHModel{T<:AbstractFloat,
	     				  d,
						  VS<:MultivariateVolatilitySpec{T, d},
						  SD<:MultivariateStandardizedDistribution{T, d},
						  MS<:MeanSpec{T}
						 } <: ARCHModel
"""
mutable struct MultivariateARCHModel{T<:AbstractFloat,
									 d,
                 				     VS<:MultivariateVolatilitySpec{T, d},
                 		  	  	     SD<:MultivariateStandardizedDistribution{T, d},
                 				     MS<:MeanSpec{T}
                 				   	} <: ARCHModel
    spec::VS
    data::Matrix{T}
    dist::SD
    meanspec::Vector{MS}
	fitted::Bool
    function MultivariateARCHModel{T, d, VS, SD, MS}(spec, data, dist, meanspec, fitted) where {T, d, VS, SD, MS}
        new(spec, data, dist, meanspec, fitted)
    end
end

function loglikelihood(am::MultivariateARCHModel)
	sigs = covariances(am)
	z = residuals(am; standardized=true, decorrelated=true)
	n, d = size(am.data)
	return -.5 * (n * d * log(2π) + sum(logdet.(cholesky.(sigs))) + sum(z.^2))
end

"""
	MultivariateARCHModel(spec::MultivariateVolatilitySpec, data::Matrix;
          			  	  dist=MultivariateStdNormal,
					  	  meanspec::[NoIntercept{T}() for _ in 1:d]
		  			  	  fitted::Bool=false
					  	  )
Create a MultivariateARCHModel.
"""
function MultivariateARCHModel(spec::VS,
							   data::Matrix{T};
          					   dist::SD=MultivariateStdNormal{T, d}(),
          				 	   meanspec::Vector{MS}=[NoIntercept{T}() for _ in 1:d], # should come up with a proper multivariate version
		  			 		   fitted::Bool=false
          					  ) where {T<:AbstractFloat,
							   		   d,
                    			 	   VS<:MultivariateVolatilitySpec{T, d},
                   					   SD<:MultivariateStandardizedDistribution,
                   					   MS<:MeanSpec
                   			 		  }
    MultivariateARCHModel{T, d, VS, SD, MS}(spec, data, dist, meanspec, fitted)
end

"""
    predict(am::MultivariateARCHModel, what=:covariance)
Form a 1-step ahead prediction from `am`. `what` controls which object is predicted.
The choices are `:covariance` (the default) or `:correlation`.
"""
function predict(am::MultivariateARCHModel; what=:covariance)
    Ht = covariances(am)
    Rt = correlations(am)
    H = uncond(am.spec)
    R = to_corr(H)
    zt = residuals(am; decorrelated=false)
    at = residuals(am; standardized=false, decorrelated=false)
	T = nobs(am)
	zt = [zt[t, :] for t in 1:T]
	at = [at[t, :] for t in 1:T]
    update!(Ht, Rt, H, R, zt, at, typeof(am.spec), coef(am.spec))
    if what == :covariance
        return Ht[end]
    elseif what == :correlation
        return Rt[end]
    else
        error("Prediction target $what unknown.")
    end
end

# documented in general
fit(am::MultivariateARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...) = fit(typeof(am.spec), am.data; dist=typeof(am.dist), meanspec=am.meanspec[1], algorithm=algorithm, autodiff=autodiff, kwargs...) # hacky. need multivariate version

# documented in general
function fit!(am::MultivariateARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)
    am2 = fit(typeof(am.spec), am.data; meanspec=am.meanspec[1], method=am.spec.method, dist=typeof(am.dist), algorithm=algorithm, autodiff=autodiff, kwargs...)
    am.spec = am2.spec
    am.dist = am2.dist
    am.meanspec = am2.meanspec
    am.fitted = true
    am
end

# documented in general
function simulate(spec::MultivariateVolatilitySpec{T2, d}, nobs;
                  warmup=100,
                  dist::MultivariateStandardizedDistribution{T2}=MultivariateStdNormal{T2, d}(),
                  meanspec::Vector{<:MeanSpec{T2}}=[NoIntercept{T2}() for i = 1:d]
                  ) where {T2<:AbstractFloat, d}
    data = zeros(T2, nobs, d)
	_simulate!(data, spec; warmup=warmup, dist=dist, meanspec=meanspec)
	return MultivariateARCHModel(spec, data; dist=dist, meanspec=meanspec, fitted=false)
end





function _simulate!(data::Matrix{T2}, spec::MultivariateVolatilitySpec{T2, d};
                  warmup=100,
                  dist::MultivariateStandardizedDistribution{T2}=MultivariateStdNormal{T2, d}(),
                  meanspec::Vector{<:MeanSpec{T2}}=[NoIntercept{T2}() for i = 1:d]
                  ) where {T2<:AbstractFloat, d}
	@assert warmup >= 0

	T, d2 = size(data)
	@assert d == d2
	simdata = zeros(T2, T + warmup, d)
	r1 = presample(typeof(spec))
	r2 = maximum(presample.(meanspec))
	r = max(r1, r2)
	r = max(r, 1) # make sure this works for, e.g., ARCH{0}; CircularBuffer requires at least a length of 1
	Ht = CircularBuffer{Matrix{T2}}(r)
	Rt = CircularBuffer{Matrix{T2}}(r)
	zt = CircularBuffer{Vector{T2}}(r)
	at = CircularBuffer{Vector{T2}}(r)

	@inbounds begin
        H = uncond(spec)
		R = to_corr(H)
		all(eigvals(H) .> 0) || error("Model is nonstationary.")
		themean = zeros(T2, d)
        for t = 1: warmup + T
			for i = 1:d
				if t > r2
					ht = getindex.(Ht, i, i)
					lht = log.(ht)
					themean[i] = mean(getindex.(at, i), ht, lht, simdata[:, i], meanspec[i], meanspec[i].coefs, t)
				else
					themean[i] = uncond(meanspec[i])
				end
			end

			if t>r1
                update!(Ht, Rt, H, R, zt, at, typeof(spec), coef(spec))
            else
				push!(Ht, H)
				push!(Rt, R)
            end

			z = rand(dist)
			push!(zt, cholesky(Rt[end], check=false).L * z)
			push!(at, sqrt.(diag(Ht[end])) .* zt[end])
			simdata[t, :] .= themean + at[end]
        end
    end
    data .= simdata[warmup + 1 : end, :]
end
