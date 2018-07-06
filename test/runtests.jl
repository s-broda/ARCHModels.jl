using Base.Test

using ARCH
#=
Data from [1]. See [2] for a comparsion of GARCH software based on this data.
[1] Bollerslev, T. and Ghysels, E. (1996), Periodic Autoregressive Conditional Heteroscedasticity, Journal of Business and Economic Statistics (14), pp. 139-151. https://doi.org/10.2307/1392425
[2] Brooks, C., Burke, S. P., and Persand, G. (2001), Benchmarks and the accuracy of GARCH model estimation, International Journal of Forecasting (17), pp. 45-56. https://doi.org/10.1016/S0169-2070(00)00070-4
=#
#using HTTP
#res=HTTP.get("http://people.stern.nyu.edu/wgreene/Text/Edition7/TableF20-1.txt")
#r=convert.(Float64, readcsv(IOBuffer(res.body))[2:end])

T = 10^4;
spec = GARCH{1, 1}([1., .9, .05])
srand(1);
data = simulate(spec, T);
srand(1);
datat = simulate(spec, T; dist=StdTDist(4))
srand(1);
datam = simulate(spec, T; dist=StdTDist(4), meanspec=Intercept(3))
ht = Float64[]
lht = Float64[]
zt = Float64[]
am = selectmodel(GARCH, data; meanspec=NoIntercept)
am2 = ARCHModel(spec, data)
fit!(am2)
am3 = fit(am2)
am4 = selectmodel(GARCH, datat; dist=StdTDist, meanspec=NoIntercept)
am5 = selectmodel(GARCH, datam; dist=StdTDist)
am6 = fit(GARCH{1, 1}, data)
srand(1)
datae = simulate(EGARCH{1, 1, 1}([.1, 0., .9, .1]), T; meanspec=Intercept(3))

am7 = selectmodel(EGARCH, datae; maxpq=2)
@test isfitted(am) == true
@test loglikelihood(ARCHModel(spec, data)) ==  ARCH.loglik!(ht, lht, zt,
                                                            typeof(spec),
                                                            StdNormal{Float64},
                                                            NoIntercept{Float64},
                                                            data,
                                                            spec.coefs
                                                            )

@test nobs(am) == T
@test dof(am) == 3
@test coefnames(GARCH{1, 1}) == ["ω", "β₁", "α₁"]
@test coefnames(am5) == ["ω", "β₁", "α₁", "ν", "μ"]
@test all(coeftable(am4).cols[2] .== stderror(am4))

@test all(isapprox.(coef(am), [0.9086632896184081,
                               0.9055268468427705,
                               0.050367854809777915], rtol=1e-4))

@test all(isapprox.(stderror(am), [0.14582381264705224,
                                   0.010354562480367474,
                                   0.005222817398477784], rtol=1e-4))

@test all(am2.spec.coefs .== am.spec.coefs)
@test all(am3.spec.coefs .== am2.spec.coefs)
@test all(isapprox(coef(am4), [0.8307014299672306,
                               0.9189503152734588,
                               0.042080807758329355,
                               3.835646488238764], rtol=1e-4))

@test all(isapprox(coef(am5), [0.8306175556436268,
                               0.9189538270625667,
                               0.04208964132482301,
                               3.8348509665880797,
                               2.9918445831618024], rtol=1e-4))

@test all(isapprox(coef(am6), [0.910496430719689,
                               0.9054120402733519,
                               0.05039127076312942,
                               0.027705636765390795], rtol=1e-4))

@test all(isapprox(coef(am7), [0.08502955535533116,
                               0.004709708474515596,
                               0.9164935566284109,
                               0.09325947325535855,
                               3.0137461089470308], rtol=1e-4))
@test coefnames(EGARCH{2, 2, 2}) == ["ω", "γ₁", "γ₂", "β₁", "β₂", "α₁", "α₂"]

@test_warn "Fisher" stderror(ARCHModel(GARCH{3, 0}([1., .1, .2, .3]), [.1, .2, .3, .4, .5, .6, .7]))

@test_warn "negative" stderror(ARCHModel(GARCH{3, 0}([1., .1, .2, .3]), -5*[.1, .2, .3, .4, .5, .6, .7]))
e = @test_throws ARCH.NumParamError ARCH.loglik!(ht, lht, zt, typeof(spec), StdNormal{Float64},
                                                 NoIntercept{Float64}, data,
                                                 [0., 0., 0., 0.]
                                                 )
str = sprint(showerror, e.value)
@test startswith(str, "incorrect number of parameters")
@test_throws ARCH.NumParamError GARCH{1, 1}([.1])
io = IOBuffer()
str = sprint(io -> show(io, am))
@test startswith(str, "\nGARCH{1,1}")
@test selectmodel(ARCH._ARCH, data).spec.coefs == fit(ARCH._ARCH{3}, data).spec.coefs
@test typeof(StdNormal())==typeof(StdNormal(Float64[]))
@test fit(StdNormal, data).coefs == Float64[]
@test coefnames(StdNormal) == String[]
@test ARCH.distname(StdNormal) == "Gaussian"
srand(1)
data = rand(StdTDist(4), 10000)
@test fit(StdTDist, data).coefs[1] ≈ 3.972437329588246 rtol=1e-4
@test coefnames(StdTDist) == ["ν"]
@test ARCH.distname(StdTDist) == "Student's t"
@test typeof(NoIntercept()) == NoIntercept{Float64}
