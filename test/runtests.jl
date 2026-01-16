using Test

using ARCHModels
using GLM
using DataFrames
using StableRNGs


T = 10^4;

@testset "lgamma" begin
    @test ARCHModels.lgamma(1.0f0) == 0.0f0
end

@testset "TGARCH" begin
    @test ARCHModels.nparams(TGARCH{1, 2, 3}) == 7
    @test ARCHModels.presample(TGARCH{1, 2, 3}) == 3
    spec = TGARCH{1,1,1}([1., .05, .9, .01]);
    str = sprint(show, spec)
    if VERSION < v"1.5.5"
        @test startswith(str, "TGARCH{1,1,1} specification.")
    else
        @test startswith(str, "TGARCH{1, 1, 1} specification.\n\n─────────────────────────────────\n               ω    γ₁   β₁    α₁\n─────────────────────────────────\nParameters:  1.0  0.05  0.9  0.01\n─────────────────────────────────\n")
    end
    am = simulate(spec, T, rng=StableRNG(1));
    am = selectmodel(TGARCH, am.data; meanspec=NoIntercept(), show_trace=true, maxlags=2)
    @test all(isapprox.(coef(am), [1.3954654215590847,
                                   0.06693040956623193,
                                   0.8680818765441008,
                                   0.006665140784151278], rtol=1e-4))
   #everything below is just pure GARCH, in fact
    spec = GARCH{1, 1}([1., .9, .05])
    am0 = simulate(spec, T; rng=StableRNG(1));
    am00 = deepcopy(am0)
    am00.data .= 0.
    simulate!(am00, rng=StableRNG(1))
    @test all(am00.data .== am0.data)
    am00 = simulate(am0; rng=StableRNG(1))
    @test all(am00.data .== am0.data)
    am000 = simulate(am0, nobs(am0); rng=StableRNG(1))
    @test all(am000.data .== am0.data)
    am = selectmodel(GARCH, am0.data; meanspec=NoIntercept(), show_trace=true)
    @test isfitted(am) == true
    @test all(isapprox.(coef(am), [1.116707484875346,
                                   0.8920705288828562,
                                   0.05103227915762242], rtol=1e-4))
    @test all(isapprox.(stderror(am), [ 0.22260057264313066,
                                        0.016030182299773734,
                                        0.006460941055580745], rtol=1e-3))
    @test sum(volatilities(am0)) ≈ 44285.00568611553
    @test sum(abs, residuals(am0)) ≈ 7964.585890843087
    @test sum(abs, residuals(am0, standardized=false)) ≈ 35281.71207401529
    am2 = UnivariateARCHModel(spec, am0.data)
    @test isfitted(am2) == false
    io = IOBuffer()
    str = sprint(io -> show(io, am2))
    if VERSION < v"1.5.5"
        @test startswith(str, "\nTGARCH{0,1,1}")
    else
        @test startswith(str, "\nGARCH{1, 1}")
    end
    fit!(am2)
    @test isfitted(am2) == true
    io = IOBuffer()
    str = sprint(io -> show(io, am2))
    if VERSION < v"1.5.5"
        @test startswith(str, "\nTGARCH{0,1,1}")
    else
        @test startswith(str, "\nGARCH{1, 1}")
    end
    am3 = fit(am2)
    @test isfitted(am3) == true
    @test all(am2.spec.coefs .== am.spec.coefs)
    @test all(am3.spec.coefs .== am2.spec.coefs)
end
@testset "ARCH" begin
    spec = ARCH{2}([1., .3, .4]);
    am = simulate(spec, T; rng=StableRNG(1));
    @test selectmodel(ARCH, am.data).spec.coefs == fit(ARCH{2}, am.data).spec.coefs
    spec = ARCH{0}([1.]);
    am = simulate(spec, T, rng=StableRNG(1));
    fit!(am)
    @test all(isapprox.(coef(am),  0.991377950108106, rtol=1e-4))

end

@testset "EGARCH" begin
    @test ARCHModels.nparams(EGARCH{1, 2, 3}) == 7
    @test ARCHModels.presample(EGARCH{1, 2, 3}) == 3
    am = simulate(EGARCH{1, 1, 1}([.1, 0., .9, .1]), T; meanspec=Intercept(3), rng=StableRNG(1))
    am7 = selectmodel(EGARCH, am.data; maxlags=2, show_trace=true)
    @test all(isapprox(coef(am7), [ 0.1240152087585493,
                                   -0.010544394266072957,
                                    0.874501604519596,
                                    0.10762246065941368,
                                    3.0008464829419053], rtol=1e-4))

    @test coefnames(EGARCH{2, 2, 2}) == ["ω", "γ₁", "γ₂", "β₁", "β₂", "α₁", "α₂"]
    @test_throws Base.ErrorException predict.(am7, :variance, 1:3)
end
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

@testset "Distributions" begin
    a=rand(StableRNG(1), StdT(3))

    b=rand(StableRNG(1), StdT(3), 1)[1]
    @test a==b

    @test rand(StableRNG(1), StdNormal()) ≈ -0.5325200748641231
    @testset "Gaussian" begin
        data = rand(StableRNG(1), T)
        @test typeof(StdNormal())==typeof(StdNormal(Float64[]))
        @test fit(StdNormal, data).coefs == Float64[]
        @test coefnames(StdNormal) == String[]
        @test ARCHModels.distname(StdNormal) == "Gaussian"
        @test quantile(StdNormal(), .05) ≈ -1.6448536269514724
        @test ARCHModels.constraints(StdNormal{Float64}, Float64) == (Float64[], Float64[])
    end
    @testset "Student" begin
        data = rand(StableRNG(1), StdT(4), T)
        spec = GARCH{1, 1}([1., .9, .05])
        @test fit(StdT, data).coefs[1] ≈ 4. atol=0.5
        @test coefnames(StdT) == ["ν"]
        @test ARCHModels.distname(StdT) == "Student's t"
        @test quantile(StdT(3), .05) ≈ -1.3587150125838563
        datat = simulate(spec, T; dist=StdT(4), rng=StableRNG(1)).data
        datam = simulate(spec, T; dist=StdT(4), meanspec=Intercept(3), rng=StableRNG(1)).data
        am4 = selectmodel(GARCH, datat; dist=StdT, meanspec=NoIntercept{Float64}(), show_trace=true)
        am5 = selectmodel(GARCH, datam; dist=StdT, show_trace=true)
        @test coefnames(am5) == ["ω", "β₁", "α₁", "ν", "μ"]
        @test all(coeftable(am4).cols[2] .== stderror(am4))
        @test isapprox(coef(am4)[4], 4., atol=0.5)
        @test isapprox(coef(am5)[4], 4., atol=0.5)
    end
    @testset "HansenSkewedT" begin
       data = rand(StableRNG(1), StdSkewT(4,-0.3), T)
       spec = GARCH{1, 1}([1., .9, .05])
       c = fit(StdSkewT, data).coefs
       @test c[1] ≈ 3.990671630456716 rtol=1e-4
       @test c[2] ≈ -0.3136773995478942 rtol=1e-4
       @test typeof(StdSkewT(3,0)) == typeof(StdSkewT(3.,0)) == typeof(StdSkewT([3,0.0]))
       @test coefnames(StdSkewT) == ["ν", "λ"]
       @test ARCHModels.nparams(StdSkewT) == 2
       @test ARCHModels.distname(StdSkewT) == "Hansen's Skewed t"
       @test ARCHModels.constraints(StdNormal{Float64}, Float64) == (Float64[], Float64[])
       @test quantile(StdSkewT(3,0), 0.5) == 0
       @test quantile(StdSkewT(3,0), .05) ≈ -1.3587150125838563
       @test ARCHModels.constraints(StdSkewT{Float64}, Float64) == (Float64[20/10, -one(Float64)], Float64[Inf,one(Float64)])
       dataskt = simulate(spec, T; dist=StdSkewT(4,-0.3), rng=StableRNG(1)).data
       datam = simulate(spec, T; dist=StdSkewT(4,-0.3), meanspec=Intercept(3), rng=StableRNG(1)).data
       am4 = selectmodel(GARCH, dataskt; dist=StdSkewT, meanspec=NoIntercept{Float64}(), show_trace=true)
       am5 = selectmodel(GARCH, datam; dist=StdSkewT, show_trace=true)
       @test coefnames(am5) == ["ω", "β₁", "α₁", "ν", "λ", "μ"]
       @test all(coeftable(am4).cols[2] .== stderror(am4))
       @test all(isapprox(coef(am4), [ 1.0123398035363282,
                                       0.9010308454299863,
                                       0.042335307040165894,
                                       4.24455990918083,
                                      -0.3115002211205442], rtol=1e-4))
       @test all(isapprox(coef(am5), [ 1.0151845148616474,
                                       0.9009908899358181,
                                       0.04243949895951436,
                                       4.241005415020919,
                                      -0.3124667515252298,
                                       2.9931917146031144], rtol=1e-4))
    end
    @testset "GED" begin
        @test typeof(StdGED(3)) == typeof(StdGED(3.)) == typeof(StdGED([3.]))
        data = rand(StableRNG(1), StdGED(1), T)
        @test fit(StdGED, data).coefs[1] ≈ 1. atol=0.5
        @test coefnames(StdGED) == ["p"]
        @test ARCHModels.nparams(StdGED) == 1
        @test ARCHModels.distname(StdGED) == "GED"
        @test quantile(StdGED(1), .05) ≈ -1.6281735335151468
    end
    @testset "Standardized" begin
        using Distributions
        @test eltype(StdNormal{Float64}()) == Float64
        MyStdT=Standardized{TDist}
        @test typeof(MyStdT([1.])) == typeof(MyStdT(1.))
        @test ARCHModels.logconst(MyStdT, [0]) == 0.
        @test coefnames(MyStdT{Float64}) == ["ν"]
        @test ARCHModels.distname(MyStdT{Float64}) == "TDist"
        @test all(isapprox.(ARCHModels.startingvals(MyStdT, [0.]), eps()))
        @test quantile(MyStdT(3.), .1) ≈ quantile(StdT(3.), .1)
        ARCHModels.startingvals(::Type{<:MyStdT}, data::Vector{T}) where T = T[3.]
        am = simulate(GARCH{1, 1}([1, 0.9, .05]), 1000, dist=MyStdT(3.); rng=StableRNG(1))
        @test  loglikelihood(fit(am)) >= -3000.
    end
end
@testset "tests" begin
    am = fit(GARCH{1, 1}, BG96)
    LM = ARCHLMTest(am)
    @test pvalue(LM) ≈ 0.1139758664282619
    str = sprint(show, LM)
    @test startswith(str, "ARCH LM test for conditional heteroskedasticity")
    @test ARCHModels.testname(LM) == "ARCH LM test for conditional heteroskedasticity"


    vars = VaRs(am, 0.01)
    DQ = DQTest(BG96, VaRs(am), 0.01)
    @test pvalue(DQ) ≈ 2.3891461144184955e-11
    str = sprint(show, DQ)
    @test startswith(str, "Engle and Manganelli's (2004) DQ test (out of sample)")
    @test ARCHModels.testname(DQ) == "Engle and Manganelli's (2004) DQ test (out of sample)"
end
@testset "multivariate" begin
    am1 = fit(DCC, DOW29[:, 1:2])
    am2 = fit(DCC, DOW29[:, 1:2]; method=:twostep)
    am3 = MultivariateARCHModel(DCC{1, 1}([1. 0.; 0. 1.], [0., 0.], [GARCH{1, 1}([1., 0., 0.]), GARCH{1, 1}([1., 0., 0.])]), DOW29[:, 1:2]) # not fitted
    am4 = fit(DCC, DOW29[1:20, 1:29]) # shrinkage n<p
    @test all(fit(am1).spec.coefs .== am1.spec.coefs)
    @test all(isapprox(am1.spec.coefs, [0.8912884521017908, 0.05515419379547665], rtol=1e-3))
    @test all(isapprox(am2.spec.coefs,    [0.8912161306136979, 0.055139392936998946], rtol=1e-3))
    @test all(isapprox(am4.spec.coefs, [0.8935938309400944, 6.938893903907228e-18], atol=1e-3))
    @test all(isapprox(stderror(am1)[1:2], [0.0434344187103969, 0.020778846682313102], rtol=1e-3))
    @test all(isapprox(stderror(am2)[1:2], [0.030405542205923865, 0.014782869078355866], rtol=1e-4))
    @test all(isapprox(predict(am1; what=:correlation)[:], [1.0, 0.4365129466277069, 0.4365129466277069, 1.0], rtol=1e-4))
    @test all(isapprox(predict(am1; what=:covariance)[:], [6.916591739333349, 1.329392154000225, 1.329392154000225,  1.340972349032465], rtol=1e-4))
    @test_throws ErrorException predict(am1; what=:bla)
    @test residuals(am1)[1, 1] ≈ 0.5107042609407892
    @test_throws ErrorException fit(DCC, DOW29; method=:bla)
    @test_throws ARCHModels.NumParamError DCC{1, 1}([1. 0.; 0. 1.], [1., 0., 0.], [GARCH{1, 1}([1., 0., 0.]), GARCH{1, 1}([1., 0., 0.])])
    @test_throws AssertionError DCC{1, 1}([1. 0.; 0. 1.], [0., 0.], [GARCH{1, 1}([1., 0., 0.]), GARCH{1, 1}([1., 0., 0.])]; method=:bla)
    @test coefnames(am1) == ["β₁", "α₁", "ω₁", "β₁₁", "α₁₁", "μ₁", "ω₂", "β₁₂", "α₁₂", "μ₂"]
    @test ARCHModels.nparams(DCC{1, 1}) == 2
    ARCHModels.nparams(DCC{1, 1, GARCH{1, 1}, Float64, 2}) == 8
    @test ARCHModels.presample(DCC{1, 2, GARCH{3, 4}}) == 4
    @test ARCHModels.presample(DCC{1, 2, GARCH{3, 4, Float64}, Float64, 2}) == 4
    io = IOBuffer()
    str = sprint(io -> show(io, am1))
    @test startswith(str, "\n2-dim")
    str = sprint(io -> show(io, am3))
    @test startswith(str, "\n2-dim")
    str = sprint(io -> show(io, am3.spec))
    @test startswith(str, "DCC{1, 1")
    str = sprint(io -> show(IOContext(io, :se=>true), am1))
    @test occursin("Std.Error", str)
    @test_throws ErrorException fit(DCC, DOW29[1:11, :]) # shrinkage requires n>=12
    @test loglikelihood(am1) ≈ -9810.905799585276

    @test ARCHModels.nparams(MultivariateStdNormal) == 0
    @test typeof(MultivariateStdNormal{Float64, 3}()) == typeof(MultivariateStdNormal{Float64, 3}(Float64[]))
    @test typeof(MultivariateStdNormal(Float64, 3)) == typeof(MultivariateStdNormal{Float64, 3}(Float64[]))
    @test typeof(MultivariateStdNormal(Float64[], 3)) == typeof(MultivariateStdNormal{Float64, 3}(Float64[]))
    @test typeof(MultivariateStdNormal{Float64}(3)) == typeof(MultivariateStdNormal{Float64, 3}(Float64[]))
    @test typeof(MultivariateStdNormal(3)) == typeof(MultivariateStdNormal{Float64, 3}(Float64[]))
    @test all(isapprox(rand(StableRNG(1), MultivariateStdNormal(2)), [-0.5325200748641231,  0.098465514284785], rtol=1e-6))
    @test coefnames(MultivariateStdNormal) == String[]
    @test ARCHModels.distname(MultivariateStdNormal) == "Multivariate Normal"


    am = am1
    am.spec.coefs .= [.7, .2]
    ams  = simulate(am; rng=StableRNG(1))
    @test isfitted(ams) == false
    fit!(ams)
    @test isfitted(ams) == true
    @test all(isapprox(ams.spec.coefs, [0.6611103068430052, 0.23089471530783906], rtol=1e-4))
    simulate!(ams; rng=StableRNG(2))
    @test ams.fitted == false
    fit!(ams)
    @test all(isapprox(ams.spec.coefs, [0.6660369039914371, 0.2329752007155509], rtol=1e-4))
    amc = fit(DCC{1, 2, GARCH{3, 2}}, DOW29[:, 1:4]; meanspec=AR{3})
    ams = simulate(amc, T; rng=StableRNG(1))
    fit!(ams)
    @test all(isapprox(ams.meanspec[1].coefs, [-0.1040426570178552, 0.03639191550146291, 0.033657970110476075, -0.020300480179225668], rtol=1e-4))
    ame = fit(DCC{1, 2, EGARCH{1, 1, 1}}, DOW29[:, 1:4])
    ams = simulate(ame, T; rng=StableRNG(1))
    fit!(ams)
    @test all(isapprox(ams.spec.univariatespecs[1].coefs, [0.05335407349997172, -0.08008165178490954,  0.9627467601623543,  0.22652855417695117], rtol=1e-4))
    ccc = fit(CCC, DOW29[:, 1:4])
    @test dof(ccc) == 16
    @test ccc.spec.R[1, 2] ≈ 0.37095654552885643
    @test isapprox(stderror(ccc)[1], 0.06298215515406534, rtol=1e-3)
    cccs = simulate(ccc, T; rng=StableRNG(1))
    @test  cccs.data[end, 1] ≈ -0.8530862593689736
    @test coefnames(ccc) == ["ω₁", "β₁₁", "α₁₁", "μ₁", "ω₂", "β₁₂", "α₁₂", "μ₂", "ω₃", "β₁₃", "α₁₃", "μ₃", "ω₄", "β₁₄", "α₁₄", "μ₄"]
    io = IOBuffer()
    str = sprint(io -> show(io, ccc))
    @test startswith(str, "\n4-dim")
    io = IOBuffer()
    str = sprint(io -> show(io, ccc.spec))
    @test startswith(str, "DCC{0, 0")
end
@testset "fixes" begin
    X = [-49.78749999996362, 2951.7375000000347, 1496.437499999923, 973.8375, 2440.662500000128, 2578.062500000019, 1064.42500000032, 3378.0625000002415, -1971.5000000001048, 4373.899999999894]
    am = fit(GARCH{2, 2}, X; meanspec = ARMA{2, 2});
    @test length(volatilities(am)) == 10
    @test isapprox(loglikelihood(am), -86.01774, rtol=.001)
    @test isapprox(predict(fit(ARMA{1, 1}, BG96), :return, 2), -0.025; atol=0.01)
end
