var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#The-ARCH-Package-1",
    "page": "Home",
    "title": "The ARCH Package",
    "category": "section",
    "text": "ARCH (Autoregressive Conditional Heteroskedasticity) models are a class of models designed to capture a feature of financial returns data known as volatility clustering, i.e., the fact that large (in absolute value) returns tend to cluster together, such as during periods of financial turmoil, which then alternate with relatively calmer periods.The basic ARCH model was introduced by Engle (1982, Econometrica, pp. 987–1008), who in 2003 was awarded a Nobel Memorial Prize in Economic Sciences for its development. Today, the most popular variant is the generalized ARCH, or GARCH, model and its various extensions, due to Bollerslev (1986, Journal of Econometrics, pp. 307 - 327). The basic GARCH(1,1) model for a sample of daily asset returns r_t_tin1ldotsT isr_t=sigma_tz_tquad z_tsimmathrmN(01)quad\nsigma_t^2=omega+alpha r_t-1^2+beta sigma_t-1^2quad omega alpha beta0quad alpha+beta1This can be extended by including additional lags of past squared returns and volatilities: the GARCH(p, q) model  has q of the former and p of the latter. Another generalization is to allow  z_t to follow other, non-Gaussian distributions.This package implements simulation, estimation, and model selection for the following models:GARCH(p, q)\nEGARCH(o, p q)As for error distributions, the user may choose among the following:Standard Normal\nStandardized Student\'s t\nStandardized Generalized Error Distribution"
},

{
    "location": "#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "The package is not yet registered. To install it in Julia 1.0 or later, doadd https://github.com/s-broda/ARCH.jlin the Pkg REPL mode (which is entered by pressing ] at the prompt). For Julia 0.6, check out the 0.6 branch."
},

{
    "location": "#Contents-1",
    "page": "Home",
    "title": "Contents",
    "category": "section",
    "text": "Pages = [\"types.md\", \"manual.md\", \"reference.md\"]\nDepth = 2"
},

{
    "location": "#Acknowledgements-1",
    "page": "Home",
    "title": "Acknowledgements",
    "category": "section",
    "text": "This project has received funding from the European Research Council (ERC) under the European Union\'s Horizon 2020 research and innovation program (grant agreement No. 750559).(Image: EU LOGO)"
},

{
    "location": "types/#",
    "page": "Introduction and type hierarchy",
    "title": "Introduction and type hierarchy",
    "category": "page",
    "text": ""
},

{
    "location": "types/#Introduction-and-type-hierarchy-1",
    "page": "Introduction and type hierarchy",
    "title": "Introduction and type hierarchy",
    "category": "section",
    "text": "Consider a sample of daily asset returns r_t_tin1ldotsT. All models covered in this package share the same basic structure, in that they decompose the return into a conditional mean and a mean-zero innovation:r_t=mu_t+sigma_tz_tquad mu_tequivmathbbEr_tmidmathcalF_t-1quad sigma_t^2equivmathbbE(r_t-mu_t)^2midmathcalF_t-1where z_t is identically and independently distributed according to some law with mean zero and unit variance and mathcalF_t is the natural filtration of r_t (i.e., it encodes information about past returns).This package represents a (G)ARCH model as an instance of ARCHModel, which implements the interface of StatisticalModel from StatsBase. An instance of this type contains a vector of data (such as equity returns), and encapsulates information about the volatility specification (e.g., GARCH or EGARCH), the mean specification (e.g., whether an intercept is included), and the error distribution."
},

{
    "location": "types/#volaspec-1",
    "page": "Introduction and type hierarchy",
    "title": "Volatility specifications",
    "category": "section",
    "text": "Volatility specifications describe the evolution of sigma_t. They are modelled as subtypes of VolatilitySpec. There is one type for each class of (G)ARCH model, parameterized by numbers of lags."
},

{
    "location": "types/#GARCH-1",
    "page": "Introduction and type hierarchy",
    "title": "GARCH",
    "category": "section",
    "text": "The GARCH(p, q) model, due to Bollerslev (1986) specifies the volatility assigma_t^2=omega+sum_i=1^pbeta_i sigma_t-i^2+sum_i=1^qalpha_i r_t-i^2 quad omega alpha_i beta_i0quad sum_i=1^max pq alpha_i+beta_i1The corresponding type is GARCH{p, q}. For example, a GARCH(1, 1) model with ω=1, β=9, and α=05 is obtained withjulia> using ARCH\n\njulia> GARCH{1, 1}([1., .9, .05])\nGARCH{1,1} specification.\n\n               ω  β₁   α₁\nParameters:  1.0 0.9 0.05As for all subtypes of VolatilitySpec, the order of the parameters in the coefficient vector is such that all parameters pertaining to the first type parameter p (corresponding to the first sum in the equation) appear before those pertaining to the second, q.As a special case, the ARCH(q) volatility specification, due to Engle (1982), issigma_t^2=omega+sum_i=1^qalpha_i r_t-i^2corresponding to a GARCH{0, q} model. It is available as _ARCH{q}:julia> _ARCH{2}([1., .5, .4])\nGARCH{0,2} specification.\n\n               ω  α₁  α₂\nParameters:  1.0 0.5 0.4"
},

{
    "location": "types/#EGARCH-1",
    "page": "Introduction and type hierarchy",
    "title": "EGARCH",
    "category": "section",
    "text": "The EGARCH{o, p, q} volatility specification, due to Nelson (1991), islog(sigma_t^2)=omega+sum_i=1^ogamma_i z_t-i+sum_i=1^pbeta_i log(sigma_t-i^2)+sum_i=1^qalpha_i (z_t-i-sqrt2pi) quad z_t=r_tsigma_tquad sum_i=1^pbeta_i1The corresponding type is EGARCH{o, p, q}:julia> EGARCH{1, 1, 1}([-0.1, .1, .9, .04])\nEGARCH{1,1,1} specification.\n\n                ω  γ₁  β₁   α₁\nParameters:  -0.1 0.1 0.9 0.04"
},

{
    "location": "types/#meanspec-1",
    "page": "Introduction and type hierarchy",
    "title": "Mean specifications",
    "category": "section",
    "text": "Mean specifications serve to specify mu_t. They are modelled as subtypes of MeanSpec. They contain their parameters as (possibly empty) vectors, but convenience constructors are provided where appropriate. Currently, two specifications are available:mu_t=0, available as NoIntercept:julia> NoIntercept() # convenience constructor, eltype defaults to Float64\nNoIntercept{Float64}(Float64[])mu_t=mu, available as Intercept:julia> Intercept(3) # convenience constructor\nIntercept{Float64}([3.0])"
},

{
    "location": "types/#Distributions-1",
    "page": "Introduction and type hierarchy",
    "title": "Distributions",
    "category": "section",
    "text": ""
},

{
    "location": "types/#Built-in-distributions-1",
    "page": "Introduction and type hierarchy",
    "title": "Built-in distributions",
    "category": "section",
    "text": "Different standardized (mean zero, variance one) distributions for z_t are available as subtypes of StandardizedDistribution. StandardizedDistribution in turn subtypes Distribution{Univariate, Continuous} from Distributions.jl, though not the entire interface must necessarily be implemented. StandardizedDistributions again hold their parameters as vectors, but convenience constructors are provided. The following are currently available:StdNormal, the standard normal distribution:julia> StdNormal() # convenience constructor\nStdNormal{Float64}(coefs=Float64[])StdT, the standardized Student\'s t distribution:julia> StdT(3) # convenience constructor\nStdT{Float64}(coefs=[3.0])StdGED, the standardized Generalized Error Distribution:julia> StdGED(1) # convenience constructor\nStdGED{Float64}(coefs=[1.0])"
},

{
    "location": "types/#User-defined-standardized-distributions-1",
    "page": "Introduction and type hierarchy",
    "title": "User-defined standardized distributions",
    "category": "section",
    "text": "Apart from the natively supported standardized distributions, it is possible to wrap a continuous univariate distribution from the Distributions package in the Standardized wrapper type. Below, we reimplement the standardized normal distribution:julia> using Distributions\n\njulia> const MyStdNormal = Standardized{Normal};MyStdNormal can be used whereever a built-in distribution could, albeit with a speed penalty. Note also that if the underlying distribution (such as Normal in the example above) contains location and/or scale parameters, then these are no longer identifiable, which implies that the estimated covariance matrix of the estimators will be singular.A final remark concerns the domain of the parameters: the estimation process relies on a starting value for the parameters of the distribution, say thetaequiv(theta_1 ldots theta_p). For a distribution wrapped in Standardized, the starting value for theta_i is taken to be a small positive value ϵ. This will fail if ϵ is not in the domain of theta_i; as an example, the standardized Student\'s t distribution is only defined for degrees of freedom larger than 2, because a finite variance is required for standardization. In that case, it is necessary to define a method of the (non-exported) function startingvals that returns a feasible vector of starting values, as follows:julia> const MyStdT = Standardized{TDist};\n\njulia> ARCH.startingvals(::Type{<:MyStdT}, data::Vector{T}) where T = T[3.]"
},

{
    "location": "types/#Working-with-ARCHModels-1",
    "page": "Introduction and type hierarchy",
    "title": "Working with ARCHModels",
    "category": "section",
    "text": "The constructor for ARCHModel takes two mandatory arguments: an instance of a subtype of VolatilitySpec, and a vector of returns. The mean specification and error distribution can be changed via the keyword arguments meanspec and dist, which respectively default to NoIntercept and StdNormal.For example, to construct a GARCH(1, 1) model with an intercept and t-distributed errors, one would dojulia> spec = GARCH{1, 1}([1., .9, .05]);\n\njulia> data = BG96;\n\njulia> am = ARCHModel(spec, data; dist=StdT(3.), meanspec=Intercept(1.))\n\nGARCH{1,1} model with Student\'s t errors, T=1974.\n\n\n                             μ\nMean equation parameters:  1.0\n\n                             ω  β₁   α₁\nVolatility parameters:     1.0 0.9 0.05\n\n                             ν\nDistribution parameters:   3.0The model can then be fitted as follows:julia> fit!(am)\n\nGARCH{1,1} model with Student\'s t errors, T=1974.\n\n\nMean equation parameters:\n\n       Estimate  Std.Error  z value Pr(>|z|)\nμ    0.00227251 0.00686802 0.330882   0.7407\n\nVolatility parameters:\n\n       Estimate  Std.Error z value Pr(>|z|)\nω    0.00232225 0.00163909 1.41679   0.1565\nβ₁     0.884488   0.036963  23.929   <1e-99\nα₁     0.124866  0.0405471 3.07952   0.0021\n\nDistribution parameters:\n\n     Estimate Std.Error z value Pr(>|z|)\nν     4.11211  0.400384 10.2704   <1e-24It should, however, rarely be necessary to construct an ARCHModel manually via its constructor; typically, instances of it are created by calling fit, selectmodel, or simulate.As discussed earlier, ARCHModel implements the interface of StatisticalModel from StatsBase, so you can call coef, coefnames, confint, dof, informationmatrix, isfitted, loglikelihood, nobs,  score, stderror, vcov, etc. on its instances:julia> nobs(am)\n1974Other useful methods include volatilities and residuals."
},

{
    "location": "manual/#",
    "page": "Usage",
    "title": "Usage",
    "category": "page",
    "text": "DocTestSetup = quote\n    using Random\n    Random.seed!(1)\nend\nDocTestFilters = r\".*[0-9\\.]\""
},

{
    "location": "manual/#Usage-1",
    "page": "Usage",
    "title": "Usage",
    "category": "section",
    "text": "We will be using the data from Bollerslev and Ghysels (1986), available as the constant BG96. The data consist of daily German mark/British pound exchange rates (1974 observations) and are often used in evaluating implementations of (G)ARCH models (see, e.g., Brooks et.al. (2001). We begin by convincing ourselves that the data exhibit ARCH effects; a quick and dirty way of doing this is to look at the sample autocorrelation function of the squared returns:julia> using ARCH\n\njulia> data = BG96;\n\njulia> autocor(data.^2, 1:10, demean=true) # re-exported from StatsBase\n10-element Array{Float64,1}:\n 0.22294073831639766\n 0.17663183540117078\n 0.14086005904595456\n 0.1263198344036979\n 0.18922204038617135\n 0.09068404029331875\n 0.08465365332525085\n 0.09671690899919724\n 0.09217329577285414\n 0.11984168975215709Using a critical value of 196sqrt1974=0044, we see that there is indeed significant autocorrelation in the squared series.A more formal test for the presence of volatility clustering is Engle\'s (1982) ARCH-LM test. The test statistic is given by LMequiv TR^2_aux, where R^2_aux is the coefficient of determination in a regression of the squared returns on an intercept and p of their own lags. The test statistic follows a chi^2_p distribution under the null of no volatility clustering.julia> ARCHLMTest(BG96, 1)\nARCH LM test for conditional heteroskedasticity\n-----------------------------------------------\nPopulation details:\n    parameter of interest:   T⋅R² in auxiliary regression of uₜ² on an intercept and its own lags\n    value under h_0:         0\n    point estimate:          98.12107516935244\n\nTest summary:\n    outcome with 95% confidence: reject h_0\n    p-value:                     <1e-22\n\nDetails:\n    sample size:                    1974\n    number of lags:                 1\n    LM statistic:                   98.12107516935244The null is strongly rejected, again providing evidence for the presence of volatility clustering."
},

{
    "location": "manual/#Estimation-1",
    "page": "Usage",
    "title": "Estimation",
    "category": "section",
    "text": "Having established the presence of volatility clustering, we can begin by fitting the workhorse model of volatility modeling, a GARCH(1, 1) with standard normal errors;  for other model classes such as EGARCH, see the section on volatility specifications.julia> fit(GARCH{1, 1}, data)\n\nGARCH{1,1} model with Gaussian errors, T=1974.\n\n\nMean equation parameters:\n\n        Estimate  Std.Error   z value Pr(>|z|)\nμ    -0.00616637 0.00920163 -0.670139   0.5028\n\nVolatility parameters:\n\n      Estimate  Std.Error z value Pr(>|z|)\nω    0.0107606 0.00649493 1.65677   0.0976\nβ₁    0.805875  0.0725003 11.1155   <1e-27\nα₁    0.153411  0.0536586 2.85903   0.0042This returns an instance of ARCHModel, as described in the section Working with ARCHModels. The parameters alpha_1 and beta_1 in the volatility equation are highly significant, again confirming the presence of volatility clustering. Note also that the fitted values are the same as those found by Bollerslev and Ghysels (1986) and Brooks et.al. (2001) for the same dataset.The fit method supports a number of keyword arguments; the full signature isfit(::Type{<:VolatilitySpec}, data::Vector; dist=StdNormal, meanspec=Intercept, algorithm=BFGS(), autodiff=:forward, kwargs...)Their meaning is as follows:dist: the error distribution. A subtype (not instance) of StandardizedDistribution; see Section Distributions.\nmeanspec=Intercept: the mean specification. A subtype of MeanSpec; see the section on mean specification.The remaining keyword arguments are passed on to the optimizer.As an example, an EGARCH(1, 1, 1) model without intercept and with  Student\'s t errors is fitted as follows:julia> fit(EGARCH{1, 1, 1}, data; meanspec=NoIntercept, dist=StdT)\n\nEGARCH{1,1,1} model with Student\'s t errors, T=1974.\n\n\nVolatility parameters:\n\n       Estimate Std.Error   z value Pr(>|z|)\nω    -0.0162014 0.0186806 -0.867286   0.3858\nγ₁   -0.0378454  0.018024  -2.09972   0.0358\nβ₁     0.977687  0.012558   77.8538   <1e-99\nα₁     0.255804 0.0625497   4.08961    <1e-4\n\nDistribution parameters:\n\n     Estimate Std.Error z value Pr(>|z|)\nν     4.12423   0.40059 10.2954   <1e-24An alternative approach to fitting a VolatilitySpec to data is to first construct an ARCHModel containing the data, and then using fit! to modify it in place:julia> am = ARCHModel(GARCH{1, 1}([1., 0., 0.]), data)\n\nGARCH{1,1} model with Gaussian errors, T=1974.\n\n\n                             ω  β₁  α₁\nVolatility parameters:     1.0 0.0 0.0\n\n\n\njulia> fit!(am)\n\nGARCH{1,1} model with Gaussian errors, T=1974.\n\n\nVolatility parameters:\n\n      Estimate  Std.Error z value Pr(>|z|)\nω    0.0108661 0.00657449 1.65277   0.0984\nβ₁    0.804431  0.0730395 11.0136   <1e-27\nα₁    0.154597  0.0539319 2.86651   0.0042Calling fit(am) will return a new instance of ARCHModel instead:julia> am2 = fit(am);\n\njulia> am2 === am\nfalse\n\njulia> am2.spec.coefs == am.spec.coefs\ntrue"
},

{
    "location": "manual/#Model-selection-1",
    "page": "Usage",
    "title": "Model selection",
    "category": "section",
    "text": "The function selectmodel can be used for automatic model selection, based on an information crititerion. Given a class of model (i.e., a subtype of VolatilitySpec), it will return a fitted ARCHModel, with the lag length parameters (i.e., p and q in the case of GARCH) chosen to minimize the desired criterion. The BIC is used by default.Eg., the following selects the optimal (minimum AIC) EGARCH(o, p, q) model, where o, p, q < 2,  assuming t distributed errors.julia> selectmodel(EGARCH, data; criterion=aic, maxlags=2, dist=StdT)\n\nEGARCH{1,1,2} model with Student\'s t errors, T=1974.\n\n\nMean equation parameters:\n\n       Estimate  Std.Error  z value Pr(>|z|)\nμ    0.00196126 0.00695292 0.282077   0.7779\n\nVolatility parameters:\n\n       Estimate Std.Error   z value Pr(>|z|)\nω    -0.0031274 0.0112456 -0.278101   0.7809\nγ₁   -0.0307681 0.0160754  -1.91398   0.0556\nβ₁     0.989056 0.0073654   134.284   <1e-99\nα₁     0.421644 0.0678139   6.21767    <1e-9\nα₂    -0.229068 0.0755326   -3.0327   0.0024\n\nDistribution parameters:\n\n     Estimate Std.Error z value Pr(>|z|)\nν     4.18795  0.418697 10.0023   <1e-22Passing the keyword argument show_trace=false will show the criterion for each model after it is estimated."
},

{
    "location": "manual/#Risk-measures-1",
    "page": "Usage",
    "title": "Risk measures",
    "category": "section",
    "text": "One of the primary uses of ARCH models is for estimating and forecasting risk measures, such as Value at Risk and Expected Shortfall. This section details the relevant functionality provided in this package.Basic in-sample estimates for the Value at Risk implied by an estimated ARCHModel can be obtained using VaRs:julia> am = fit(GARCH{1, 1}, BG96);\n\njulia> VaRs(am)[end]\n0.7945179524273573"
},

{
    "location": "manual/#Forecasting-1",
    "page": "Usage",
    "title": "Forecasting",
    "category": "section",
    "text": "The predict(am::ARCHModel) method can be used to construct one-step ahead forecasts for a number of quantities. Its signature is    predict(am::ARCHModel, what=:volatility; level=0.01)The keyword argument what controls which object is predicted; the choices are :volatility (the default), :variance, :return, and :VaR. The VaR level can be controlled with the keyword argument level."
},

{
    "location": "manual/#Model-diagnostics-and-specification-tests-1",
    "page": "Usage",
    "title": "Model diagnostics and specification tests",
    "category": "section",
    "text": "Testing volatility models in general relies on the estimated conditional volatilities hatsigma_t and the standardized residuals hatz_tequiv (r_t-hatmu_t)hatsigma_t, accessible via volatilities(::ARCHModel) and residuals(::ARCHModel), respectively. The non-standardized residuals hatu_tequiv r_t-hatmu_t can be obtained by passing standardized=false as a keyword argument to residuals.One possibility to test a volatility specification is to apply the ARCH-LM test to the standardized residuals. This is achieved by calling ARCHLMTest on the estimated ARCHModel:julia> am = fit(GARCH{1, 1}, BG96);\n\njulia> ARCHLMTest(am, 4) # 4 lags in test regression.\nARCH LM test for conditional heteroskedasticity\n-----------------------------------------------\nPopulation details:\n    parameter of interest:   T⋅R² in auxiliary regression of uₜ² on an intercept and its own lags\n    value under h_0:         0\n    point estimate:          4.211230445141555\n\nTest summary:\n    outcome with 95% confidence: fail to reject h_0\n    p-value:                     0.3782\n\nDetails:\n    sample size:                    1974\n    number of lags:                 4\n    LM statistic:                   4.211230445141555By default, the number of lags is chosen as the maximum order of the volatility specification (e.g., max(p q) for a GARCH(p, q) model). Here, the test does not reject, indicating that a GARCH(1, 1) specification is sufficient for modelling the volatility clustering (a common finding)."
},

{
    "location": "manual/#Simulation-1",
    "page": "Usage",
    "title": "Simulation",
    "category": "section",
    "text": "To simulate from an ARCHModel, use simulate. You can either specify the VolatilitySpec (and optionally the distribution and mean specification) and desired number of observations, or pass an existing ARCHModel. Use simulate! to modify the data in place.julia> am3 = simulate(GARCH{1, 1}([1., .9, .05]), 1000; warmup=500, meanspec=Intercept(5.), dist=StdT(3.))\n\nGARCH{1,1} model with Student\'s t errors, T=1000.\n\n\n                             μ\nMean equation parameters:  5.0\n\n                             ω  β₁   α₁\nVolatility parameters:     1.0 0.9 0.05\n\n                             ν\nDistribution parameters:   3.0\n\njulia> am4 = simulate(am3, 1000); # passing the number of observations is optional, the default being nobs(am3)DocTestSetup = nothing\nDocTestFilters = nothing"
},

{
    "location": "reference/#",
    "page": "Reference guide",
    "title": "Reference guide",
    "category": "page",
    "text": ""
},

{
    "location": "reference/#Reference-guide-1",
    "page": "Reference guide",
    "title": "Reference guide",
    "category": "section",
    "text": ""
},

{
    "location": "reference/#Index-1",
    "page": "Reference guide",
    "title": "Index",
    "category": "section",
    "text": ""
},

{
    "location": "reference/#ARCH.ARCH",
    "page": "Reference guide",
    "title": "ARCH.ARCH",
    "category": "module",
    "text": "The ARCH package for Julia. For documentation, see https://s-broda.github.io/ARCH.jl/dev.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.BG96",
    "page": "Reference guide",
    "title": "ARCH.BG96",
    "category": "constant",
    "text": "BG96\n\nData from Bollerslev and Ghysels (JBES 1996).\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.ARCHLMTest",
    "page": "Reference guide",
    "title": "ARCH.ARCHLMTest",
    "category": "type",
    "text": "ARCHLMTest <: HypothesisTest\n\nEngle\'s (1982) LM test for autoregressive conditional heteroskedasticity.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.ARCHLMTest",
    "page": "Reference guide",
    "title": "ARCH.ARCHLMTest",
    "category": "type",
    "text": "ARCHLMTest(am::ARCHModel, p=max(o, p, q, ...))\n\nConduct Engle\'s (1982) LM test for autoregressive conditional heteroskedasticity with p lags in the test regression.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.ARCHLMTest-Union{Tuple{T}, Tuple{Array{T,1},Integer}} where T<:Real",
    "page": "Reference guide",
    "title": "ARCH.ARCHLMTest",
    "category": "method",
    "text": "ARCHLMTest(u::Vector, p::Integer)\n\nConduct Engle\'s (1982) LM test for autoregressive conditional heteroskedasticity with p lags in the test regression.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.ARCHModel",
    "page": "Reference guide",
    "title": "ARCH.ARCHModel",
    "category": "type",
    "text": "ARCHModel{T<:AbstractFloat,\n          VS<:VolatilitySpec,\n          SD<:StandardizedDistribution{T},\n          MS<:MeanSpec{T}\n          } <: StatisticalModel\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.ARCHModel-Union{Tuple{MS}, Tuple{SD}, Tuple{VS}, Tuple{T}, Tuple{VS,Array{T,1}}} where MS<:MeanSpec where SD<:StandardizedDistribution where VS<:VolatilitySpec where T<:AbstractFloat",
    "page": "Reference guide",
    "title": "ARCH.ARCHModel",
    "category": "method",
    "text": "ARCHModel(spec::VolatilitySpec, data::Vector; dist=StdNormal(),\n          meanspec=NoIntercept(), fitted=false\n          )\n\nCreate an ARCHModel.\n\nExample:\n\njulia> ARCHModel(GARCH{1, 1}([1., .9, .05]), randn(10))\n\nGARCH{1,1} model with Gaussian errors, T=10.\n\n\n                             ω  β₁   α₁\nVolatility parameters:     1.0 0.9 0.05\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.EGARCH",
    "page": "Reference guide",
    "title": "ARCH.EGARCH",
    "category": "type",
    "text": "EGARCH{o, p, q, T<:AbstractFloat} <: VolatilitySpec{T}\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.EGARCH-Union{Tuple{Array{T,1}}, Tuple{T}, Tuple{q}, Tuple{p}, Tuple{o}} where T where q where p where o",
    "page": "Reference guide",
    "title": "ARCH.EGARCH",
    "category": "method",
    "text": "EGARCH{o, p, q}(coefs) -> VolatilitySpec\n\nConstruct an EGARCH specification with the given parameters.\n\nExample:\n\njulia> EGARCH{1, 1, 1}([-0.1, .1, .9, .04])\nEGARCH{1,1,1} specification.\n\n                ω  γ₁  β₁   α₁\nParameters:  -0.1 0.1 0.9 0.04\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.GARCH",
    "page": "Reference guide",
    "title": "ARCH.GARCH",
    "category": "type",
    "text": "GARCH{p, q, T<:AbstractFloat} <: VolatilitySpec{T}\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.GARCH-Union{Tuple{Array{T,1}}, Tuple{T}, Tuple{q}, Tuple{p}} where T where q where p",
    "page": "Reference guide",
    "title": "ARCH.GARCH",
    "category": "method",
    "text": "GARCH{p, q}(coefs) -> VolatilitySpec\n\nConstruct a GARCH specification with the given parameters.\n\nExample:\n\njulia> GARCH{2, 1}([1., .3, .4, .05 ])\nGARCH{2,1} specification.\n\n               ω  β₁  β₂   α₁\nParameters:  1.0 0.3 0.4 0.05\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.Intercept",
    "page": "Reference guide",
    "title": "ARCH.Intercept",
    "category": "type",
    "text": "Intercept{T} <: MeanSpec{T}\n\nA mean specification with just an intercept.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.Intercept-Tuple{Any}",
    "page": "Reference guide",
    "title": "ARCH.Intercept",
    "category": "method",
    "text": "Intercept(mu)\n\nCreate an instance of Intercept. mu can be passed as a scalar or vector.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.MeanSpec",
    "page": "Reference guide",
    "title": "ARCH.MeanSpec",
    "category": "type",
    "text": "MeanSpec{T}\n\nAbstract supertype that mean specifications inherit from.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.NoIntercept",
    "page": "Reference guide",
    "title": "ARCH.NoIntercept",
    "category": "type",
    "text": "NoIntercept{T} <: MeanSpec{T}\n\nA mean specification without an intercept (i.e., the mean is zero).\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.NoIntercept",
    "page": "Reference guide",
    "title": "ARCH.NoIntercept",
    "category": "type",
    "text": "NoIntercept(T::Type=Float64)\nNoIntercept{T}()\nNoIntercept(v::Vector)\n\nCreate an instance of NoIntercept.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.Standardized",
    "page": "Reference guide",
    "title": "ARCH.Standardized",
    "category": "type",
    "text": "Standardized{D<:ContinuousUnivariateDistribution, T}  <: StandardizedDistribution{T}\n\nA wrapper type for standardizing a distribution from Distributions.jl.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.StandardizedDistribution",
    "page": "Reference guide",
    "title": "ARCH.StandardizedDistribution",
    "category": "type",
    "text": "StandardizedDistribution{T} <: Distributions.Distribution{Univariate, Continuous}\n\nAbstract supertype that standardized distributions inherit from.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.StdGED",
    "page": "Reference guide",
    "title": "ARCH.StdGED",
    "category": "type",
    "text": "StdGED{T} <: StandardizedDistribution{T}\n\nThe standardized (mean zero, variance one) generalized error distribution.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.StdGED-Tuple{Any}",
    "page": "Reference guide",
    "title": "ARCH.StdGED",
    "category": "method",
    "text": "StdGED(p)\n\nCreate a standardized generalized error distribution parameter p. p can be passed as a scalar or vector.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.StdNormal",
    "page": "Reference guide",
    "title": "ARCH.StdNormal",
    "category": "type",
    "text": "StdNormal{T} <: StandardizedDistribution{T}\n\nThe standard Normal distribution.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.StdNormal",
    "page": "Reference guide",
    "title": "ARCH.StdNormal",
    "category": "type",
    "text": "StdNormal(T::Type=Float64)\nStdNormal(v::Vector)\nStdNormal{T}()\n\nConstruct an instance of StdNormal.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.StdT",
    "page": "Reference guide",
    "title": "ARCH.StdT",
    "category": "type",
    "text": "StdT{T} <: StandardizedDistribution{T}\n\nThe standardized (mean zero, variance one) Student\'s t distribution.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.StdT-Tuple{Any}",
    "page": "Reference guide",
    "title": "ARCH.StdT",
    "category": "method",
    "text": "StdT(v)\n\nCreate a standardized t distribution with v degrees of freedom. ν` can be passed as a scalar or vector.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.VolatilitySpec",
    "page": "Reference guide",
    "title": "ARCH.VolatilitySpec",
    "category": "type",
    "text": "VolatilitySpec{T}\n\nAbstract supertype that volatility specifications inherit from.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH._ARCH",
    "page": "Reference guide",
    "title": "ARCH._ARCH",
    "category": "type",
    "text": "_ARCH{q, T<:AbstractFloat} <: VolatilitySpec{T}\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.VaRs",
    "page": "Reference guide",
    "title": "ARCH.VaRs",
    "category": "function",
    "text": "VaRs(am::ARCHModel, level=0.01)\n\nReturn the in-sample Value at Risk implied by am.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.selectmodel-Union{Tuple{MS}, Tuple{SD}, Tuple{T}, Tuple{VS}, Tuple{Type{VS},Array{T,1}}} where MS<:MeanSpec where SD<:StandardizedDistribution where T<:AbstractFloat where VS<:VolatilitySpec",
    "page": "Reference guide",
    "title": "ARCH.selectmodel",
    "category": "method",
    "text": "selectmodel(::Type{VS}, data; kwargs...) -> ARCHModel\n\nFit the volatility specification VS with varying lag lengths and return that which minimizes the BIC.\n\nKeyword arguments:\n\ndist=StdNormal: the error distribution.\nmeanspec=Intercept: the mean specification.\nmaxlags=3: maximum lag length to try in each parameter of VS.\ncriterion=bic: function that takes an ARCHModel and returns the criterion to minimize.\nshow_trace=false: print criterion to screen for each estimated model.\nalgorithm=BFGS(), autodiff=:forward, kwargs...: passed on to the optimizer.\n\nExample\n\njulia> selectmodel(EGARCH, BG96)\n\nEGARCH{1,1,2} model with Gaussian errors, T=1974.\n\n\nMean equation parameters:\n\n        Estimate  Std.Error   z value Pr(>|z|)\nμ    -0.00900018 0.00943948 -0.953461   0.3404\n\nVolatility parameters:\n\n       Estimate Std.Error   z value Pr(>|z|)\nω    -0.0544398 0.0592073 -0.919478   0.3578\nγ₁   -0.0243368 0.0270414 -0.899985   0.3681\nβ₁     0.960301 0.0388183   24.7384   <1e-99\nα₁     0.405788  0.067466    6.0147    <1e-8\nα₂    -0.207357  0.114161  -1.81636   0.0693\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.simulate",
    "page": "Reference guide",
    "title": "ARCH.simulate",
    "category": "function",
    "text": "simulate(am::ARCHModel; warmup=100)\nsimulate(am::ARCHModel, nobs; warmup=100)\nsimulate(spec::VolatilitySpec, nobs; warmup=100, dist=StdNormal(), meanspec=NoIntercept())\n\nSimulate an ARCHModel.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.simulate!-Tuple{ARCHModel}",
    "page": "Reference guide",
    "title": "ARCH.simulate!",
    "category": "method",
    "text": "simulate!(am::ARCHModel; warmup=100)\n\nSimulate an ARCHModel, modifying am in place.\n\n\n\n\n\n"
},

{
    "location": "reference/#ARCH.volatilities-Union{Tuple{ARCHModel{T,VS,SD,MS}}, Tuple{MS}, Tuple{SD}, Tuple{VS}, Tuple{T}} where MS where SD where VS where T",
    "page": "Reference guide",
    "title": "ARCH.volatilities",
    "category": "method",
    "text": "volatilities(am::ARCHModel)\n\nReturn the conditional volatilities.\n\n\n\n\n\n"
},

{
    "location": "reference/#Statistics.mean-Tuple{MeanSpec}",
    "page": "Reference guide",
    "title": "Statistics.mean",
    "category": "method",
    "text": "mean(spec::MeanSpec)\n\nReturn the mean implied by MeanSpec\n\n\n\n\n\n"
},

{
    "location": "reference/#StatsBase.fit!-Tuple{ARCHModel}",
    "page": "Reference guide",
    "title": "StatsBase.fit!",
    "category": "method",
    "text": "fit!(am::ARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)\n\nFit the ARCHModel specified by am, modifying am in place. Keyword arguments are passed on to the optimizer.\n\n\n\n\n\n"
},

{
    "location": "reference/#StatsBase.fit-Tuple{ARCHModel}",
    "page": "Reference guide",
    "title": "StatsBase.fit",
    "category": "method",
    "text": "fit(am::ARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)\n\nFit the ARCHModel specified by am and return the result in a new instance of ARCHModel. Keyword arguments are passed on to the optimizer.\n\n\n\n\n\n"
},

{
    "location": "reference/#StatsBase.fit-Union{Tuple{T}, Tuple{MS}, Tuple{SD}, Tuple{VS}, Tuple{Type{VS},Array{T,1}}} where T<:AbstractFloat where MS<:MeanSpec where SD<:StandardizedDistribution where VS<:VolatilitySpec",
    "page": "Reference guide",
    "title": "StatsBase.fit",
    "category": "method",
    "text": "fit(VS::Type{<:VolatilitySpec}, data; dist=StdNormal, meanspec=Intercept,\n    algorithm=BFGS(), autodiff=:forward, kwargs...)\n\nFit the ARCH model specified by VS to data.\n\nKeyword arguments:\n\ndist=StdNormal: the error distribution.\nmeanspec=Intercept: the mean specification.\nalgorithm=BFGS(), autodiff=:forward, kwargs...: passed on to the optimizer.\n\nExample: EGARCH{1, 1, 1} model without intercept, Student\'s t errors.\n\njulia> fit(EGARCH{1, 1, 1}, BG96; meanspec=NoIntercept, dist=StdT)\n\nEGARCH{1,1,1} model with Student\'s t errors, T=1974.\n\n\nVolatility parameters:\n\n       Estimate Std.Error   z value Pr(>|z|)\nω    -0.0162014 0.0186806 -0.867286   0.3858\nγ₁   -0.0378454  0.018024  -2.09972   0.0358\nβ₁     0.977687  0.012558   77.8538   <1e-99\nα₁     0.255804 0.0625497   4.08961    <1e-4\n\nDistribution parameters:\n\n     Estimate Std.Error z value Pr(>|z|)\nν     4.12423   0.40059 10.2954   <1e-24\n\n\n\n\n\n"
},

{
    "location": "reference/#StatsBase.fit-Union{Tuple{T}, Tuple{SD}, Tuple{Type{SD},Array{T,1}}} where T<:AbstractFloat where SD<:StandardizedDistribution",
    "page": "Reference guide",
    "title": "StatsBase.fit",
    "category": "method",
    "text": "fit(::Type{SD}, data; algorithm=BFGS(), kwargs...)\n\nFit a standardized distribution to the data, using the MLE. Keyword arguments are passed on to the optimizer.\n\n\n\n\n\n"
},

{
    "location": "reference/#StatsBase.predict-Union{Tuple{ARCHModel{T,VS,SD,MS}}, Tuple{MS}, Tuple{SD}, Tuple{VS}, Tuple{T}, Tuple{ARCHModel{T,VS,SD,MS},Any}} where MS where SD where VS where T",
    "page": "Reference guide",
    "title": "StatsBase.predict",
    "category": "method",
    "text": "predict(am::ARCHModel, what=:volatility; level=0.01)\n\nForm a 1-step ahead prediction from am. what controls which object is predicted. The choices are :volatility (the default), :variance, :return, and :VaR. The VaR level can be controlled with the keyword argument level.\n\n\n\n\n\n"
},

{
    "location": "reference/#StatsBase.residuals-Union{Tuple{ARCHModel{T,VS,SD,MS}}, Tuple{MS}, Tuple{SD}, Tuple{VS}, Tuple{T}} where MS where SD where VS where T",
    "page": "Reference guide",
    "title": "StatsBase.residuals",
    "category": "method",
    "text": "residuals(am::ARCHModel; standardized=true)\n\nReturn the residuals of the model. Pass standardized=false for the non-devolatized residuals.\n\n\n\n\n\n"
},

{
    "location": "reference/#Public-API-1",
    "page": "Reference guide",
    "title": "Public API",
    "category": "section",
    "text": "DocTestSetup = quote\n    using ARCH\n    using Random\n    Random.seed!(1)\nend\nDocTestFilters = r\".*[0-9\\.]\"Modules = [ARCH]\nPrivate = falseDocTestSetup = nothing\nDocTestFilters = nothing"
},

]}
