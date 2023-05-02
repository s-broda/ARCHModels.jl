#Todo:
#HAC s.e.s from CovariancesMatrices.jl?
#Float16/32 don't seem to work anymore. Problem in Optim?
#support missing data? timeseries?
#implement lrtest
#allow uninititalized constructors for UnivariateVolatilitySpec, MeanSpec and StandardizedDistribution? If so, then be consistent with how they are defined
#  (change for meanspec and dist ), document, and test. Also, NaN is prob. safer than undef.
#logconst needs to return the correct type
"""
The ARCHModels package for Julia. For documentation, see https://s-broda.github.io/ARCHModels.jl/dev.
"""
module ARCHModels
using Reexport
@reexport using StatsBase
using StatsFuns: normcdf, normccdf, normlogpdf, norminvcdf, log2π, logtwo, RFunctions.tdistinvcdf, RFunctions.gammainvcdf
using GLM: modelmatrix, response, LinearModel
using SpecialFunctions: beta, gamma, digamma #, lgamma
using MuladdMacro
using PrecompileTools
# work around https://github.com/JuliaMath/SpecialFunctions.jl/issues/186
# until https://github.com/JuliaDiff/ForwardDiff.jl/pull/419/ is merged
# remove test in runtests.jl as well when this gets fixed
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
import Random: rand, AbstractRNG, GLOBAL_RNG
import HypothesisTests: HypothesisTest, testname, population_param_of_interest, default_tail, show_params, pvalue
import StatsBase: StatisticalModel, stderror, loglikelihood, nobs, fit, fit!, confint, aic,
                  bic, aicc, dof, coef, coefnames, coeftable, CoefTable,
				  informationmatrix, islinear, score, vcov, residuals, predict
import StatsModels: TableRegressionModel
export ARCHModel, UnivariateARCHModel, UnivariateVolatilitySpec, StandardizedDistribution, Standardized, MeanSpec,
       simulate, simulate!, selectmodel, StdNormal, StdT, StdGED, StdSkewT, Intercept, Regression,
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
@static if VERSION >= v"1.9.0-alpha1"
	@compile_workload begin
		io = IOBuffer()
		se = stderr
		redirect_stderr()
		# autocor(BG96.^2, 1:4, demean=true)
		# m = selectmodel(TGARCH, BG96)
		# show(io, m)
		m = fit(GARCH{1, 1}, BG96)
		show(io, m)
		m = fit(EGARCH{1, 1, 1}, BG96)
		show(io, m)
		
		# m = fit(GARCH{1, 1}, BG96; dist=StdSkewT)
		# show(io, m)
		# m = fit(GARCH{1, 1}, BG96; dist=StdGED)
		#show(io, m)
		ARCHLMTest(m, 4)
		# vars = VaRs(m, 0.05)
		# predict(m, :volatility; level=0.01)
		# t = DQTest([1., 2.], [.1, .1], .01)
		# show(io, t)
		# m = selectmodel(EGARCH, BG96)
		# show(io, m)
		# m = selectmodel(ARMA, BG96)
		# show(io, m)
		# m = fit(DCC, DOW29)
		# show(io, m)
		# simulate(GARCH{1, 1}([1., .9, .05]), 1000; warmup=500, meanspec=Intercept(5.), dist=StdT(3.))
		redirect_stderr(se)
	end # precompile block
end # if
end # module
