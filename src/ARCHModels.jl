#Todo:
#plotting via timeseries
#PkgBenchmark
#HAC s.e.s from CovariancesMatrices.jl?
#how to export arch?
#Forecasting
#Float16/32 don't seem to work anymore. Problem in Optim?
#support missing data? timeseries?
#a simulated AM should probably contain a (zero) intercept, so that fit! is consistent with fit.
#the constructor for UnivariateARCHModel should make a copy of its args
#implement lrtest
#allow uninititalized constructors for VolatilitySpec, MeanSpec and StandardizedDistribution? If so, then be consistent with how they are defined
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
import Random: rand, AbstractRNG
import HypothesisTests: HypothesisTest, testname, population_param_of_interest, default_tail, show_params, pvalue
import StatsBase: StatisticalModel, stderror, loglikelihood, nobs, fit, fit!, confint, aic,
                  bic, aicc, dof, coef, coefnames, coeftable, CoefTable,
				  informationmatrix, islinear, score, vcov, residuals, predict
export ARCHModel, UnivariateARCHModel, VolatilitySpec, StandardizedDistribution, Standardized, MeanSpec,
       simulate, simulate!, selectmodel, StdNormal, StdT, StdGED, Intercept, Regression,
       NoIntercept, ARMA, AR, MA, BG96, volatilities, mean, quantile, VaRs, pvalue, means


include("utils.jl")
include("general.jl")
include("univariatearchmodel.jl")
include("meanspecs.jl")
include("univariatestandardizeddistributions.jl")
include("EGARCH.jl")
include("TGARCH.jl")
include("tests.jl")
function __init__()
	@require GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a" begin
		using .GLM
		import .StatsModels: DataFrameRegressionModel
		function fit(vs::Type{VS}, lm::DataFrameRegressionModel{<:LinearModel}; kwargs...) where VS<:VolatilitySpec
			fit(vs, response(lm.model); meanspec=Regression(modelmatrix(lm.model); coefnames=coefnames(lm)), kwargs...)
		end
		function fit(vs::Type{VS}, lm::LinearModel; kwargs...) where VS<:VolatilitySpec
			fit(vs, response(lm); meanspec=Regression(modelmatrix(lm)), kwargs...)
		end
	end
end
end#module
