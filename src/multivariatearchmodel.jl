
const DOW29 = readdlm(joinpath(dirname(pathof(ARCHModels)), "data", "dow29.csv"), ',')

abstract type MultivariateStandardizedDistribution{T, d} <: Distribution{Multivariate, Continuous} end

abstract type MultivariateVolatilitySpec{T, d} end

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

fit(am::MultivariateARCHModel; kwargs...) = fit(typeof(am.spec), am.data; dist=typeof(am.dist), meanspec=am.meanspec[1], kwargs...) # hacky. need multivariate version

function loglikelihood(am::MultivariateARCHModel)
	sigs = covariances(am)
	z = residuals(am; standardized=true, decorrelated=true)
	n, d = size(am.data)
	return -.5 * (n * d * log(2π) + sum(logdet.(cholesky.(sigs))) + sum(z.^2))
end

function MultivariateARCHModel(spec::VS,
							   data::Matrix{T},
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
    analytical_shrinkage(X::Matrix)
Analytical nonlinear shrinkage estimator of the covariance matrix. Based on the
Matlab code from [1]. Translated to Julia and used here under MIT license by
permission from the authors.

[1] Ledoit, O., and Wolf, M. (2018), "Analytical Nonlinear Shrinkage of
Large-Dimensional Covariance Matrices", University of Zurich Econ WP 264.
https://www.econ.uzh.ch/static/workingpapers_iframe.php?id=943
"""
function analytical_shrinkage(X)
n, p = size(X)
@assert n >= 12 # important: sample size n must be >= 12
sample = Symmetric(X'*X) / n
E = eigen(sample)
lambda = E.values
u = E.vectors

# compute analytical nonlinear shrinkage kernel formula
lambda = lambda[max(1, p-n+1):p]
L = repeat(lambda, 1, min(p, n))
h = n^(-1/3) # Equation (4.9)
H = h*L'
x = (L-L') ./ H
ftilde = (3/4/sqrt(5)) * mean(max.(1 .- x.^2 ./ 5, 0) ./ H, dims=2) # Equation (4.7)
Hftemp = (-3/10/pi) * x + (3/4/sqrt(5)/pi) * (1 .- x.^2 ./ 5) .* log.(abs.((sqrt(5).-x) ./ (sqrt(5).+x))) # Equation (4.8)
Hftemp[abs.(x) .== sqrt(5)] .= (-3/10/pi) .* x[abs.(x) .== sqrt(5)]
Hftilde = mean(Hftemp./H, dims=2)
if p<=n
    dtilde = lambda ./ ((pi * (p/n) *lambda .* ftilde).^2  + (1 .- (p/n) .- pi * (p/n) * lambda .* Hftilde).^2) # Equation (4.3)
else
    Hftilde0 = (1/pi) * (3/10/h^2 + 3/4/sqrt(5)/h*(1-1/5/h^2) * log((1+sqrt(5)*h)/(1-sqrt(5)*h)))*mean(1 ./ lambda) # Equation (C.8)
    dtilde0 = 1 / (pi * (p-n) / n * Hftilde0) # Equation (C.5)
    dtilde1 = lambda ./ (pi^2*lambda.^2 .* (ftilde.^2 + Hftilde.^2)) # Eq. (C.4)
    dtilde = [dtilde0*ones(p-n,1); dtilde1]
end
return u * Diagonal(dtilde[:]) * u'
end

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

function simulate(spec::MultivariateVolatilitySpec{T2, d}, nobs;
                  warmup=100,
                  dist::MultivariateStandardizedDistribution{T2}=MultivariateStdNormal{T2, d}(),
                  meanspec::Vector{<:MeanSpec{T2}}=[NoIntercept{T2}() for i = 1:d]
                  ) where {T2<:AbstractFloat, d}
    data = zeros(T2, nobs, d)
	_simulate!(data, spec; warmup=warmup, dist=dist, meanspec=meanspec)
	return data
end

function simulate(am::MultivariateARCHModel; warmup=100)
	am2 = deepcopy(am)
	simulate!(am2; warmup=warmup)
	am2.fitted=false
	am2
end
function simulate!(am::MultivariateARCHModel; warmup=100)
	_simulate!(am.data, am.spec; warmup=warmup, dist=am.dist, meanspec=am.meanspec)
	am.fitted = false
	am
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
	#r2 = presample(meanspec[1])
	#r = max(r1, r2)
	r = r1
	Ht = CircularBuffer{Array{T2, d}}(r)
	Rt = CircularBuffer{Array{T2, d}}(r)
	zt = CircularBuffer{Vector{T2}}(r)
	at = CircularBuffer{Vector{T2}}(r)

	r = max(r, 1) # make sure this works for, e.g., ARCH{0}; CircularBuffer requires at least a length of 1
    @inbounds begin
        H = uncond(spec)
		R = to_corr(H)
		all(eigvals(H) .> 0) || error("Model is nonstationary.")
        for t = 1: warmup + T
			if t>r1
                update!(Ht, Rt, H, R, zt, at, typeof(spec), coef(spec))
            else
				push!(Ht, H)
				push!(Rt, R)
            end
			z = rand(dist)
			push!(zt, cholesky(Rt[end], check=false).L * z)
			push!(at, sqrt.(diag(Ht[end])) .* zt[end])
			simdata[t, :] .= at[end]
        end
    end
    data .= simdata[warmup + 1 : end, :]
end
