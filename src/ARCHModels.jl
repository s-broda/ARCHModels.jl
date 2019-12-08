#Todo:
#PkgBenchmark
#HAC s.e.s from CovariancesMatrices.jl?
#Float16/32 don't seem to work anymore. Problem in Optim?
#support missing data? timeseries?
#a simulated AM should probably contain a (zero) intercept, so that fit! is consistent with fit.
#the constructor for UnivariateARCHModel should make a copy of its args
#implement lrtest
#allow uninititalized constructors for UnivariateVolatilitySpec, MeanSpec and StandardizedDistribution? If so, then be consistent with how they are defined
#  (change for meanspec and dist ), document, and test. Also, NaN is prob. safer than undef.
#logconst needs to return the correct type
"""
The ARCHModels package for Julia. For documentation, see https://s-broda.github.io/ARCHModels.jl/dev.
"""
module ARCHModels
using Reexport
using Requires
@reexport using StatsBase
using StatsFuns: normcdf, normccdf, normlogpdf, norminvcdf, log2π, logtwo, RFunctions.tdistrand, RFunctions.tdistinvcdf, RFunctions.gammarand, RFunctions.gammainvcdf
using SpecialFunctions: beta, gamma, digamma, # lgamma

# work around https://github.com/JuliaMath/SpecialFunctions.jl/issues/186
# until https://github.com/JuliaDiff/ForwardDiff.jl/pull/419/ is merged
using Base.Math: libm
using ForwardDiff: Dual, value, partials
@inline lgamma(x::Float64) = ccall((:lgamma, libm), Float64, (Float64,), x)
@inline lgamma(x::Float32) = ccall((:lgammaf, libm), Float32, (Float32,), x)
@inline lgamma(d::Dual{T}) where T = Dual{T}(lgamma(value(d)), digamma(value(d)) * partials(d))


using Optim
using ForwardDiff
using Distributions
using HypothesisTests
using Roots
using LinearAlgebra
using DataStructures: CircularBuffer
using DelimitedFiles
using Statistics: cov

import Distributions: quantile
import Base: show, showerror, eltype
import Statistics: mean
import Random: rand, AbstractRNG
import HypothesisTests: HypothesisTest, testname, population_param_of_interest, default_tail, show_params, pvalue
import StatsBase: StatisticalModel, stderror, loglikelihood, nobs, fit, fit!, confint, aic,
                  bic, aicc, dof, coef, coefnames, coeftable, CoefTable,
				  informationmatrix, islinear, score, vcov, residuals, predict
export ARCHModel, UnivariateARCHModel, UnivariateVolatilitySpec, StandardizedDistribution, Standardized, MeanSpec,
       simulate, simulate!, selectmodel, StdNormal, StdT, StdGED, Intercept, Regression,
       NoIntercept, ARMA, AR, MA, BG96, volatilities, mean, quantile, VaRs, pvalue, means, VolatilitySpec,
	   MultivariateVolatilitySpec, MultivariateStandardizedDistribution, MultivariateARCHModel, MultivariateStdNormal,
	   EGARCH, ARCH, GARCH, TGARCH, ARCHLMTest, DQTest,
	   DOW29, DCC, CCC, covariances, correlations


include("utils.jl")
include("general.jl")
include("univariatearchmodel.jl")
include("meanspecs.jl")
include("univariatestandardizeddistributions.jl")
include("EGARCH.jl")
include("TGARCH.jl")
include("tests.jl")
include("multivariatearchmodel.jl")
include("multivariatestandardizeddistributions.jl")
include("DCC.jl")
function __init__()
	@require GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a" begin
		using .GLM
		import .StatsModels: TableRegressionModel
		function fit(vs::Type{VS}, lm::TableRegressionModel{<:LinearModel}; kwargs...) where VS<:UnivariateVolatilitySpec
			fit(vs, response(lm.model); meanspec=Regression(modelmatrix(lm.model); coefnames=coefnames(lm)), kwargs...)
		end
		function fit(vs::Type{VS}, lm::LinearModel; kwargs...) where VS<:UnivariateVolatilitySpec
			fit(vs, response(lm); meanspec=Regression(modelmatrix(lm)), kwargs...)
		end
	end
end
end#module
