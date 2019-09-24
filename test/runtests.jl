using Test

using ARCHModels
using Random
using GLM
using DataFrames

T = 10^4;

@testset "TGARCH" begin
    @test ARCHModels.nparams(TGARCH{1, 2, 3}) == 7
    @test ARCHModels.presample(TGARCH{1, 2, 3}) == 3
    Random.seed!(1)
    spec = TGARCH{1,1,1}([1., .05, .9, .01]);
    str = sprint(show, spec)
    @test startswith(str, "TGARCH{1,1,1} specification.\n\n─────────────────────────────────\n               ω    γ₁   β₁    α₁\n─────────────────────────────────\nParameters:  1.0  0.05  0.9  0.01\n─────────────────────────────────\n")
    am = simulate(spec, T);
    am = selectmodel(TGARCH, am.data; meanspec=NoIntercept(), show_trace=true, maxlags=2)
    @test all(isapprox.(coef(am), [0.9439667311150648
                                   0.04573706835008625
                                   0.9043902283152758
                                   0.012555948398277313], rtol=1e-4))
   #everything below is just pure GARCH, in fact
    Random.seed!(1)
    spec = GARCH{1, 1}([1., .9, .05])
    am0 = simulate(spec, T);
    am00 = deepcopy(am0)
    Random.seed!(1)
    am00.data .= 0.
    simulate!(am00)
    @test all(am00.data .== am0.data)
    Random.seed!(1)
    am00 = simulate(am0)
    @test all(am00.data .== am0.data)
    Random.seed!(1)
    am000 = simulate(am0, nobs(am0))
    @test all(am000.data .== am0.data)
    am = selectmodel(GARCH, am0.data; meanspec=NoIntercept(), show_trace=true)
    @test isfitted(am) == true
    #with unconditional as presample:
    #@test all(isapprox.(coef(am), [0.9086632896184081,
    #                               0.9055268468427705,
    #                               0.050367854809777915], rtol=1e-4))
    @test all(isapprox.(coef(am), [0.9086479266110243,
                                   0.905531642067773,
                                   0.050324600594884535], rtol=1e-4))
    #with unconditional as presample:
    #@test all(isapprox.(stderror(am), [0.14582381264705224,
    #                                   0.010354562480367474,
    #                                   0.005222817398477784], rtol=1e-4))
    @test all(isapprox.(stderror(am), [0.14578736059501485,
                                       0.010356284482676704,
                                       0.005228247833454602], rtol=1e-4))
    @test sum(volatilities(am0)) ≈ 44768.17421580251
    @test sum(abs, residuals(am0)) ≈ 8022.163087384836
    @test sum(abs, residuals(am0, standardized=false)) ≈ 35939.07066637026
    am2 = UnivariateARCHModel(spec, am0.data)
    @test isfitted(am2) == false
    io = IOBuffer()
    str = sprint(io -> show(io, am2))
    @test startswith(str, "\nTGARCH{0,1,1}")
    fit!(am2)
    @test isfitted(am2) == true
    io = IOBuffer()
    str = sprint(io -> show(io, am2))
    @test startswith(str, "\nTGARCH{0,1,1}")
    am3 = fit(am2)
    @test isfitted(am3) == true
    @test all(am2.spec.coefs .== am.spec.coefs)
    @test all(am3.spec.coefs .== am2.spec.coefs)
end
@testset "ARCH" begin
    Random.seed!(1);
    spec = ARCH{2}([1., .3, .4]);
    am = simulate(spec, T);
    @test selectmodel(ARCH, am.data).spec.coefs == fit(ARCH{2}, am.data).spec.coefs
    Random.seed!(1);
    spec = ARCH{0}([1.]);
    am = simulate(spec, T);
    fit!(am)
    @test all(isapprox.(coef(am), 1.013031276122647, rtol=1e-4))

end

@testset "EGARCH" begin
    Random.seed!(1)
    @test ARCHModels.nparams(EGARCH{1, 2, 3}) == 7
    @test ARCHModels.presample(EGARCH{1, 2, 3}) == 3
    am = simulate(EGARCH{1, 1, 1}([.1, 0., .9, .1]), T; meanspec=Intercept(3))
    am7 = selectmodel(EGARCH, am.data; maxlags=2, show_trace=true)
    #with unconditional as presample:
    #@test all(isapprox(coef(am7), [0.08502955535533116,
    #                               0.004709708474515596,
    #                               0.9164935566284109,
    #                               0.09325947325535855,
    #                               3.0137461089470308], rtol=1e-4))
    @test all(isapprox(coef(am7), [0.08504883253172882,
                                   0.0047015720582706125,
                                   0.9164488571272553,
                                   0.09323297680588628,
                                   3.013732273404755], rtol=1e-4))

    @test coefnames(EGARCH{2, 2, 2}) == ["ω", "γ₁", "γ₂", "β₁", "β₂", "α₁", "α₂"]
end
@testset "StatisticalModel" begin
    #not implemented: adjr2, deviance, mss, nulldeviance, r2, rss, weights
    Random.seed!(1);
    spec = GARCH{1, 1}([1., .9, .05])
    am = simulate(spec, T)
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
    #with unconditional as presample:
    #@test aic(am) ≈ 58369.082969298106 rtol=1e-4
    #@test bic(am) ≈ 58390.713990414035 rtol=1e-4
    #@test aicc(am) ≈ 58369.085370258494 rtol=1e-4
    @test aic(am) ≈ 58369.08203915834 rtol=1e-4
    @test bic(am) ≈ 58390.71306027427 rtol=1e-4
    @test aicc(am) ≈ 58369.08444011873 rtol=1e-4

    @test all(coef(am) .== am.spec.coefs)
    #with unconditional as presample:
    #@test all(isapprox(confint(am), [0.6228537382166024 1.1944723245606814;
    #                                0.8852323068993577 0.9258214298904262;
    #                                0.040131313548448275 0.060604377407810675],
    #                   rtol=1e-4)
    #                   )
    @test all(isapprox(confint(am), [0.6229099504436408 1.1943859027784078;
                                     0.8852336974680757 0.9258295866674704;
                                     0.04007742313906393 0.06057177805070514],
                       rtol=1e-4)
                       )
    #with unconditional as presample:
    #@test all(isapprox(informationmatrix(am; expected=false), [0.15326216336912968 2.9536982257433135 2.618124940552642;
    #                                                           2.9536982257433135 58.956837321202826 53.74888605159925;
    #                                                           2.618124940552642 53.74888605159925 53.29656483617587],
    #                   rtol=1e-4)
    #                   )
    @test all(isapprox(informationmatrix(am; expected=false), [0.15267905577531846 2.9430054037767617 2.607992890001237;
                                                               2.9430054037767617 58.76705213893014 53.57462949757515;
                                                               2.607992890001237 53.57462949757517 53.14231838982629],
                       rtol=1e-4)
                       )
    @test_throws ErrorException informationmatrix(am)
    #with unconditional as presample:
    #@test all(isapprox(score(am), [-4.091261171623728e-6 3.524550271549742e-5 -6.989366926291041e-5], rtol=1e-4))
    @test all(isapprox(score(am), [0. 0. 0.], atol=1e-3))
    @test islinear(am::UnivariateARCHModel) == false
    @test predict(am) ≈ 4.361606730361275
    @test predict(am, :variance) ≈ 19.023613270332767
    @test predict(am, :return) == 0.0
    @test predict(am, :VaR) ≈ 10.146614544578197
end

@testset "MeanSpecs" begin
    Random.seed!(1);
    spec = GARCH{1, 1}([1., .9, .05])
    am = simulate(spec, T; meanspec=Intercept(0.))
    fit!(am)
    #with unconditional as presample:
    #@test all(isapprox(coef(am), [0.910496430719689,
    #                               0.9054120402733519,
    #                               0.05039127076312942,
    #                               0.027705636765390795], rtol=1e-4))
    @test all(isapprox(coef(am), [0.9104793904880137,
                                  0.9054169985282575,
                                  0.05034724930058784,
                                  0.027707836268720806], rtol=1e-4))
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
    Random.seed!(1)
    spec = GARCH{1, 1}([1., .9, .05])
    am = simulate(spec, T; meanspec=ms)
    fit!(am)
    @test all(isapprox(coef(am), [0.9063506916409171,
                                  0.905682443482137,
                                  0.05021834228447521,
                                  1.079454022288992,
                                  0.45097800554911116,
                                  0.2357782619617334,
                                  -0.05909354030019596,
                                  0.2878312346045116], rtol=1e-4))
    @test predict(am, :return) ≈ -1.4572460532296017 rtol = 1e-6
    am = selectmodel(ARCH, BG96;  meanspec=AR, maxlags=2);
    @test all(isapprox(coef(am), [0.11916340875306261,
                                  0.3156862868133263,
                                  0.18331803992622006,
                                  -0.00685700871019875,
                                  0.0358362785070197], rtol=1e-4))
    @test typeof(Regression([1 2; 3 4])) == Regression{2, Float64}
    @test typeof(Regression([1. 2.; 3. 4.])) == Regression{2, Float64}
    @test typeof(Regression{Float32}([1 2; 3 4])) == Regression{2, Float32}
    @test typeof(Regression([1 2; 3 4])) == Regression{2, Float64}
    @test typeof(Regression([1, 2], [1 2; 3 4.0f0])) ==  Regression{2, Float32}
    @test typeof(Regression([1, 2.], [1 2; 3 4.0f0])) ==  Regression{2, Float64}
    @test typeof(Regression([1], [1, 2, 3, 4.0f0])) ==  Regression{1, Float32}
    @test typeof(Regression([1, 2, 3, 4.0f0])) ==  Regression{1, Float32}
    @test ARCHModels.nparams(Regression{2, Float64}) == 2

    Random.seed!(1)
    beta = [1, 2]
    reg = Regression(beta, rand(2000, 2))
    u = randn(2000)*.1
    y = reg.X*reg.coefs+u
    @test ARCHModels.coefnames(reg) == ["β₀", "β₁"]
    @test ARCHModels.presample(reg) == 0
    @test ARCHModels.constraints(typeof(reg), Float64) == ([-Inf, -Inf], [Inf, Inf])
    @test all(isapprox(ARCHModels.startingvals(reg, y),
        [1.0129824114578263, 1.9885835817762578], rtol=1e-4))
    @test ARCHModels.uncond(reg) === 0.
    Random.seed!(1)
    am = simulate(GARCH{1, 1}([1., .9, .05]), 2000; meanspec=reg, warmup=0)
    fit!(am)
    @test_throws Base.ErrorException predict(am, :return)

    @test all(isapprox(coef(am), [1.5240432453558923,
                                 0.869016093356202,
                                 0.06125683693937313,
                                 1.1773425168044198,
                                 1.7290964605805756], rtol=1e-4))
    Random.seed!(1)
    am = simulate(GARCH{1, 1}([1., .9, .05]), 1999; meanspec=reg, warmup=0)
    @test predict(am, :return) ≈ 1.2174653422550268
    data = DataFrame(X=ones(1974), Y=BG96)
    model = lm(@formula(Y ~ -1 + X), data)
    am = fit(GARCH{1, 1}, model)
    @test all(isapprox(coef(am), coef(fit(GARCH{1, 1}, BG96, meanspec=Intercept)), rtol=1e-4))
    @test coefnames(am)[end] == "X"
    @test all(isapprox(coef(am), coef(fit(GARCH{1, 1}, model.model)), rtol=1e-4))
end

@testset "VaR" begin
    am = fit(GARCH{1, 1}, BG96)
    @test sum(VaRs(am)) ≈ 2077.0976454790807
end
@testset "Errors" begin
    #with unconditional as presample:
    #@test_warn "Fisher" stderror(UnivariateARCHModel(GARCH{3, 0}([1., .1, .2, .3]), [.1, .2, .3, .4, .5, .6, .7]))
    @test_logs (:warn, "Fisher information is singular; vcov matrix is inaccurate.") stderror(UnivariateARCHModel(GARCH{1, 0}( [1.0, .1]), [0., 1.]))
    #with unconditional as presample:
    #@test_warn "non-positive" stderror(UnivariateARCHModel(GARCH{3, 0}([1., .1, .2, .3]), -5*[.1, .2, .3, .4, .5, .6, .7]))
    @test_logs (:warn, "non-positive variance encountered; vcov matrix is inaccurate.") stderror(UnivariateARCHModel(GARCH{1, 0}( [1.0, .1]), [1., 1.]))
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
    @test_throws ARCHModels.NumParamError StdGED([1., 2.])
    @test_throws ARCHModels.NumParamError Regression([1], [1 2; 3 4])
    at = zeros(10)
    data = rand(10)
    reg = Regression(data[1:5])
    @test_throws ErrorException mean(at, at, at, data, reg, [0.], 6)
end

@testset "Distributions" begin
    Random.seed!(1)
    a=rand(StdT(3))
    Random.seed!(1)
    b=rand(StdT(3), 1)[1]
    @test a==b # https://github.com/JuliaStats/Distributions.jl/issues/846
    Random.seed!(1)
    @test rand(MersenneTwister(), StdNormal()) ≈ 0.2972879845354616
    @testset "Gaussian" begin
        Random.seed!(1)
        data = rand(T)
        @test typeof(StdNormal())==typeof(StdNormal(Float64[]))
        @test fit(StdNormal, data).coefs == Float64[]
        @test coefnames(StdNormal) == String[]
        @test ARCHModels.distname(StdNormal) == "Gaussian"
        @test quantile(StdNormal(), .05) ≈ -1.6448536269514724
        @test ARCHModels.constraints(StdNormal{Float64}, Float64) == (Float64[], Float64[])
    end
    @testset "Student" begin
        Random.seed!(1)
        data = rand(StdT(4), T)
        spec = GARCH{1, 1}([1., .9, .05])
        @test fit(StdT, data).coefs[1] ≈ 3.972437329588246 rtol=1e-4
        @test coefnames(StdT) == ["ν"]
        @test ARCHModels.distname(StdT) == "Student's t"
        @test quantile(StdT(3), .05) ≈ -1.3587150125838563
        Random.seed!(1);
        datat = simulate(spec, T; dist=StdT(4)).data
        Random.seed!(1);
        datam = simulate(spec, T; dist=StdT(4), meanspec=Intercept(3)).data
        am4 = selectmodel(GARCH, datat; dist=StdT, meanspec=NoIntercept{Float64}(), show_trace=true)
        am5 = selectmodel(GARCH, datam; dist=StdT, show_trace=true)
        @test coefnames(am5) == ["ω", "β₁", "α₁", "ν", "μ"]
        @test all(coeftable(am4).cols[2] .== stderror(am4))
        #with unconditional as presample:
        #@test all(isapprox(coef(am4), [0.8307014299672306,
        #                               0.9189503152734588,
        #                               0.042080807758329355,
        #                               3.835646488238764], rtol=1e-4))
        @test all(isapprox(coef(am4), [0.831327902922751,
                                       0.9189082073187017,
                                       0.04211450420515991,
                                       3.834812845564172], rtol=1e-4))
        #with unconditional as presample:
        #@test all(isapprox(coef(am5), [0.8306175556436268,
        #                               0.9189538270625667,
        #                               0.04208964132482301,
        #                               3.8348509665880797,
        #                               2.9918445831618024], rtol=1e-4))
        @test all(isapprox(coef(am5), [0.8312482931785029,
                                       0.9189112458594789,
                                       0.04212329204579697,
                                       3.834033118707376,
                                       2.9918481871096083], rtol=1e-4))
    end
    @testset "GED" begin
        Random.seed!(1)
        @test typeof(StdGED(3)) == typeof(StdGED(3.)) == typeof(StdGED([3.]))
        data = rand(StdGED(1), T)
        @test fit(StdGED, data).coefs[1] ≈ 1.0193004687300224 rtol=1e-4
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
        Random.seed!(1)
        am = simulate(GARCH{1, 1}([1, 0.9, .05]), 1000, dist=MyStdT(3.))
        @test  loglikelihood(fit(am)) ≈ -2700.9089012063323
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
    @test all(isapprox(am1.spec.coefs, [0.8912884521017908, 0.05515419379547665], rtol=1e-4))
    @test all(isapprox(am2.spec.coefs,    [0.8912161306136979, 0.055139392936998946], rtol=1e-4))
    @test all(isapprox(am4.spec.coefs, [0.8935938309400944, 6.938893903907228e-18], rtol=1e-4))
    @test all(isapprox(stderror(am1)[1:2], [0.0434344187103969, 0.020778846682313102], rtol=1e-4))
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
    Random.seed!(1)
    @test all(isapprox(rand(MultivariateStdNormal(2)), [0.2972879845354616, 0.3823959677906078], rtol=1e-6))
    @test coefnames(MultivariateStdNormal) == String[]
    @test ARCHModels.distname(MultivariateStdNormal) == "Multivariate Normal"

    Random.seed!(1)
    am = am1
    am.spec.coefs .= [.7, .2]
    ams  = simulate(am)
    @test isfitted(ams) == false
    fit!(ams)
    @test isfitted(ams) == true
    @test all(isapprox(ams.spec.coefs, [0.7122483516102956, 0.19156028421431875], rtol=1e-4))
    Random.seed!(2)
    simulate!(ams)
    @test ams.fitted == false
    fit!(ams)
    @test all(isapprox(ams.spec.coefs, [0.6630049669013613, 0.22885770926598498], rtol=1e-4))
    Random.seed!(1)
    amc = fit(DCC{1, 2, GARCH{3, 2}}, DOW29[:, 1:4]; meanspec=AR{3})
    ams = simulate(amc, T)
    fit!(ams)
    @test all(isapprox(ams.meanspec[1].coefs, [-0.09394176323811071, 0.05159711352207107, 0.011348428433666473, -0.020175913191330077], rtol=1e-4))
    Random.seed!(1)
    ame = fit(DCC{1, 2, EGARCH{1, 1, 1}}, DOW29[:, 1:4])
    ams = simulate(ame, T)
    fit!(ams)
    @test all(isapprox(ams.spec.univariatespecs[1].coefs, [0.046700648921491957, -0.07258062140595305, 0.9664860227494515, 0.22051944930571496], rtol=1e-4))

    Random.seed!(1)
    ccc = fit(CCC, DOW29[:, 1:4])
    @test ccc.spec.R[1, 2] ≈ 0.37095654552885643
    @test stderror(ccc)[1] ≈ 0.06298215515406534
    cccs = simulate(ccc, T)
    @test  cccs.data[end, 1] ≈ -1.5061782364569236
    @test coefnames(ccc) == ["ω₁", "β₁₁", "α₁₁", "μ₁", "ω₂", "β₁₂", "α₁₂", "μ₂", "ω₃", "β₁₃", "α₁₃", "μ₃", "ω₄", "β₁₄", "α₁₄", "μ₄"]
    io = IOBuffer()
    str = sprint(io -> show(io, ccc))
    @test startswith(str, "\n4-dim")
    io = IOBuffer()
    str = sprint(io -> show(io, ccc.spec))
    @test startswith(str, "DCC{0, 0")
end
