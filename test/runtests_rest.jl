@testset "StatisticalModel" begin
    #not implemented: adjr2, deviance, mss, nulldeviance, r2, rss, weights
    spec = GARCH{1, 1}([1., .9, .05])
    am = simulate(spec, T; rng=StableRNG(1))
    fit!(am)
    @test loglikelihood(am) ==  ARCHModels.loglik!(Float64[],
                                                                Float64[],
                                                                Float64[],
                                                                Float64[],
                                                                typeof(spec),
                                                                StdNormal{Float64},
                                                                NoIntercept{Float64}(),
                                                                am.data,
                                                                spec.coefs
                                                                )
    @test nobs(am) == T
    @test dof(am) == 3
    @test coefnames(GARCH{1, 1}) == ["ω", "β₁", "α₁"]
    @test aic(am) ≈ 57949.19500673284 rtol=1e-4
    @test bic(am) ≈ 57970.82602784877 rtol=1e-4
    @test aicc(am) ≈ 57949.19740769323 rtol=1e-4

    @test all(coef(am) .== am.spec.coefs)
    @test all(isapprox(confint(am), [ 0.680418   1.553;
                                      0.860652   0.923489;
                                      0.0383691  0.0636955],
                       rtol=1e-4)
                       )
    @test all(isapprox(informationmatrix(am; expected=false)/T, [ 0.125032   2.33319   2.07012;
                                                                2.33319   44.6399   40.8553;
                                                                2.07012   40.8553   41.2192],
                       rtol=1e-4)
                       )
    @test_throws ErrorException informationmatrix(am)
    @test all(isapprox(score(am), [0. 0. 0.], atol=1e-3))
    @test islinear(am::UnivariateARCHModel) == false
    @test predict(am) ≈ 4.296827552671104
    @test predict(am, :variance) ≈ 18.46272701739355
    @test predict(am, :return) == 0.0
    @test predict(am, :VaR) ≈ 9.995915642276554
    for what in [:return, :variance]
        @test predict.(am, what, 1:3) == [predict(am, what, h) for h in 1:3]
    end
    @test_throws Base.ErrorException predict.(am, :VaR, 1:3)
    @test_throws Base.ErrorException predict.(am, :volatility, 1:3)
end

@testset "MeanSpecs" begin
    spec = GARCH{1, 1}([1., .9, .05])
    am = simulate(spec, T; meanspec=Intercept(0.), rng=StableRNG(1))
    fit!(am)
    @test all(isapprox(coef(am), [1.1176635890968043,
                                  0.8919906787166815,
                                  0.05106346071866704,
                                  0.00952591461710004], rtol=1e-4))
    @test ARCHModels.coefnames(Intercept(0.)) == ["μ"]
    @test ARCHModels.nparams(Intercept) == 1
    @test ARCHModels.presample(Intercept(0.)) == 0
    @test ARCHModels.constraints(Intercept{Float64}, Float64) == (-Float64[Inf], Float64[Inf])
    @test typeof(NoIntercept()) == NoIntercept{Float64}
    @test ARCHModels.coefnames(NoIntercept()) == []
    @test ARCHModels.constraints(NoIntercept{Float64}, Float64) == (Float64[], Float64[])
    @test ARCHModels.nparams(NoIntercept) == 0
    @test ARCHModels.presample(NoIntercept()) == 0
    @test ARCHModels.uncond(NoIntercept()) == 0
    @test mean(zeros(5), zeros(5), zeros(5), zeros(5), NoIntercept(), zeros(5), 4) == 0.
    ms = ARMA{2, 2}([1., .5, .2, -.1, .3])
    @test ARCHModels.nparams(typeof(ms)) == length(ms.coefs)
    @test ARCHModels.presample(ms) == 2
    @test ARCHModels.coefnames(ms) == ["c", "φ₁", "φ₂", "θ₁", "θ₂"]
    spec = GARCH{1, 1}([1., .9, .05])
    am = simulate(spec, T; meanspec=ms, rng=StableRNG(1))
    fit!(am)
    @test all(isapprox(coef(am), [ 1.1375727511714622,
                                   0.8903853180079492,
                                   0.05158067874765809,
                                   1.0091192373639755,
                                   0.482666588367849,
                                   0.21802258440272837,
                                  -0.08390300941364812,
                                   0.28868236034111855], rtol=1e-4))
    @test predict(am, :return) ≈ 2.335436537249963 rtol = 1e-6
    am = selectmodel(ARCH, BG96;  meanspec=AR, maxlags=2);
    @test all(isapprox(coef(am), [0.1191634087516343,
                                  0.31568628680702837,
                                  0.18331803992648235,
                                 -0.006857008709781168,
                                  0.035836278501164005], rtol=1e-4))
    @test typeof(Regression([1 2; 3 4])) == Regression{2, Float64}
    @test typeof(Regression([1. 2.; 3. 4.])) == Regression{2, Float64}
    @test typeof(Regression{Float32}([1 2; 3 4])) == Regression{2, Float32}
    @test typeof(Regression([1 2; 3 4])) == Regression{2, Float64}
    @test typeof(Regression([1, 2], [1 2; 3 4.0f0])) ==  Regression{2, Float32}
    @test typeof(Regression([1, 2.], [1 2; 3 4.0f0])) ==  Regression{2, Float64}
    @test typeof(Regression([1], [1, 2, 3, 4.0f0])) ==  Regression{1, Float32}
    @test typeof(Regression([1, 2, 3, 4.0f0])) ==  Regression{1, Float32}
    @test ARCHModels.nparams(Regression{2, Float64}) == 2

    rng = StableRNG(1)
    beta = [1, 2]
    reg = Regression(beta, rand(rng, 2000, 2))
    u = randn(rng, 2000)*.1
    y = reg.X*reg.coefs+u
    @test ARCHModels.coefnames(reg) == ["β₀", "β₁"]
    @test ARCHModels.presample(reg) == 0
    @test ARCHModels.constraints(typeof(reg), Float64) == ([-Inf, -Inf], [Inf, Inf])
    @test all(isapprox(ARCHModels.startingvals(reg, y),
        [0.992361089980835, 2.003646964507331], rtol=1e-4))
    @test ARCHModels.uncond(reg) === 0.
    am = simulate(GARCH{1, 1}([1., .9, .05]), 2000; meanspec=reg, warmup=0, rng=StableRNG(1))
    fit!(am)
    @test_throws Base.ErrorException predict(am, :return)

    @test all(isapprox(coef(am), [1.098632569628791,
                                  0.8866288812154145,
                                  0.05770241980639491,
                                  0.7697476790102007,
                                  2.403750061921962], rtol=1e-4))
    am = simulate(GARCH{1, 1}([1., .9, .05]), 1999; meanspec=reg, warmup=0, rng=StableRNG(1))
    @test predict(am, :return) ≈ 2.3760239544958175
    data = DataFrame(X=ones(1974), Y=BG96)
    model = lm(@formula(Y ~ -1 + X), data)
    am = fit(GARCH{1, 1}, model)
    @test all(isapprox(coef(am), coef(fit(GARCH{1, 1}, BG96, meanspec=Intercept)), rtol=1e-4))
    @test coefnames(am)[end] == "X"
    @test all(isapprox(coef(am), coef(fit(GARCH{1, 1}, model.model)), rtol=1e-4))
    @test sum(coef(fit(ARMA{1, 1}, BG96))) ≈ 0.21595383060382695
    @test isapprox(sum(coef(selectmodel(ARMA, BG96; minlags=2, maxlags=3))), 0.254; atol=0.1)
end

@testset "VaR" begin
    am = fit(GARCH{1, 1}, BG96)
    @test sum(VaRs(am)) ≈ 2077.0976454790807
end
@testset "Errors" begin
    #with unconditional as presample:
    #@test_warn "Fisher" stderror(UnivariateARCHModel(GARCH{3, 0}([1., .1, .2, .3]), [.1, .2, .3, .4, .5, .6, .7]))
    #@test_warn "non-positive" stderror(UnivariateARCHModel(GARCH{3, 0}([1., .1, .2, .3]), -5*[.1, .2, .3, .4, .5, .6, .7]))

    # the following are temporarily disabled while we use FiniteDiff for Hessians:
    #@test_logs (:warn, "Fisher information is singular; vcov matrix is inaccurate.") stderror(UnivariateARCHModel(GARCH{1, 0}( [1.0, .1]), [0., 1.]))
    #@test_logs (:warn, "non-positive variance encountered; vcov matrix is inaccurate.") stderror(UnivariateARCHModel(GARCH{1, 0}( [1.0, .1]), [1., 1.]))
    e = @test_throws ARCHModels.NumParamError ARCHModels.loglik!(Float64[], Float64[], Float64[], Float64[], GARCH{1, 1}, StdNormal{Float64},
                                                     NoIntercept{Float64}(), zeros(T),
                                                     [0., 0., 0., 0.]
                                                     )
    str = sprint(showerror, e.value)
    @test startswith(str, "incorrect number of parameters")
    @test_throws ARCHModels.NumParamError GARCH{1, 1}([.1])
    e = @test_throws ErrorException predict(UnivariateARCHModel(GARCH{0, 0}([1.]), zeros(10)), :blah)
    str = sprint(showerror, e.value)
    @test startswith(str, "Prediction target blah unknown")
    @test_throws ARCHModels.NumParamError ARMA{1, 1}([1.])
    @test_throws ARCHModels.NumParamError Intercept([1., 2.])
    @test_throws ARCHModels.NumParamError NoIntercept([1.])
    @test_throws ARCHModels.NumParamError StdNormal([1.])
    @test_throws ARCHModels.NumParamError StdT([1., 2.])
    @test_throws ARCHModels.NumParamError StdSkewT([2.])
    @test_throws ARCHModels.NumParamError StdGED([1., 2.])
    @test_throws ARCHModels.NumParamError Regression([1], [1 2; 3 4])
    at = zeros(10)
    data = rand(StableRNG(1), 10)
    reg = Regression(data[1:5])
    @test_throws ErrorException mean(at, at, at, data, reg, [0.], 6)
end

include("runtests_more.jl")
