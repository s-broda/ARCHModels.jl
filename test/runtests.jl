#=
using Test

using ARCH
using Random
T = 10^4;
@testset "GARCH" begin
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
    am = selectmodel(GARCH, am0.data; meanspec=NoIntercept, show_trace=true)
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
    am2 = ARCHModel(spec, am0.data)
    @test isfitted(am2) == false
    io = IOBuffer()
    str = sprint(io -> show(io, am2))
    @test startswith(str, "\nGARCH{1,1}")
    fit!(am2)
    @test isfitted(am2) == true
    io = IOBuffer()
    str = sprint(io -> show(io, am2))
    @test startswith(str, "\nGARCH{1,1}")
    am3 = fit(am2)
    @test isfitted(am3) == true
    @test all(am2.spec.coefs .== am.spec.coefs)
    @test all(am3.spec.coefs .== am2.spec.coefs)
end

@testset "StatisticalModel" begin
    #not implemented: adjr2, deviance, mss, nulldeviance, r2, rss, weights
    Random.seed!(1);
    spec = GARCH{1, 1}([1., .9, .05])
    am = simulate(spec, T)
    fit!(am)
    @test loglikelihood(am) ==  ARCH.loglik!(Float64[],
                                                                Float64[],
                                                                Float64[],
                                                                typeof(spec),
                                                                StdNormal{Float64},
                                                                NoIntercept{Float64},
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
    @test islinear(am::ARCHModel) == false
    @test predict(am) ≈ 4.366619452770822
    @test predict(am, :variance) ≈ 19.067365445316554
    @test predict(am, :return) == 0.0
    @test predict(am, :VaR) ≈ 10.158275880698804
end

@testset "MeanSpecs" begin
    mean(NoIntercept()) == 0.0
    mean(Intercept(3)) == 3.0
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

    @test typeof(NoIntercept()) == NoIntercept{Float64}
end

@testset "ARCH" begin
    Random.seed!(1);
    spec = _ARCH{2}([1., .3, .4]);
    am = simulate(spec, T);
    @test selectmodel(_ARCH, am.data).spec.coefs == fit(_ARCH{2}, am.data).spec.coefs
end

@testset "EGARCH" begin
    Random.seed!(1)
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

@testset "VaR" begin
    am = fit(GARCH{1, 1}, BG96)
    @test sum(VaRs(am)) ≈ 2077.0976454790807
end
@testset "Errors" begin
    #with unconditional as presample:
    #@test_warn "Fisher" stderror(ARCHModel(GARCH{3, 0}([1., .1, .2, .3]), [.1, .2, .3, .4, .5, .6, .7]))
    @test_logs (:warn, "Fisher information is singular; vcov matrix is inaccurate.") stderror(ARCHModel(GARCH{1, 0}( [1.0, .1]), [0., 1.]))
    #with unconditional as presample:
    #@test_warn "non-positive" stderror(ARCHModel(GARCH{3, 0}([1., .1, .2, .3]), -5*[.1, .2, .3, .4, .5, .6, .7]))
    @test_logs (:warn, "non-positive variance encountered; vcov matrix is inaccurate.") stderror(ARCHModel(GARCH{1, 0}( [1.0, .1]), [1., 1.]))
    e = @test_throws ARCH.NumParamError ARCH.loglik!(Float64[], Float64[], Float64[], GARCH{1, 1}, StdNormal{Float64},
                                                     NoIntercept{Float64}, zeros(T),
                                                     [0., 0., 0., 0.]
                                                     )
    str = sprint(showerror, e.value)
    @test startswith(str, "incorrect number of parameters")
    @test_throws ARCH.NumParamError GARCH{1, 1}([.1])
    e = @test_throws ErrorException predict(ARCHModel(GARCH{0, 0}([1.]), zeros(10)), :blah)
    str = sprint(showerror, e.value)
    @test startswith(str, "Prediction target blah unknown")

end

@testset "Distributions" begin
    @testset "Gaussian" begin
        Random.seed!(1)
        data = rand(T)
        @test typeof(StdNormal())==typeof(StdNormal(Float64[]))
        @test fit(StdNormal, data).coefs == Float64[]
        @test coefnames(StdNormal) == String[]
        @test ARCH.distname(StdNormal) == "Gaussian"
        @test quantile(StdNormal(), .05) ≈ -1.6448536269514724
    end
    @testset "Student" begin
        Random.seed!(1)
        data = rand(StdTDist(4), T)
        spec = GARCH{1, 1}([1., .9, .05])
        @test fit(StdTDist, data).coefs[1] ≈ 3.972437329588246 rtol=1e-4
        @test coefnames(StdTDist) == ["ν"]
        @test ARCH.distname(StdTDist) == "Student's t"
        @test quantile(StdTDist(3), .05) ≈ -2.3533634348018255
        Random.seed!(1);
        datat = simulate(spec, T; dist=StdTDist(4)).data
        Random.seed!(1);
        datam = simulate(spec, T; dist=StdTDist(4), meanspec=Intercept(3)).data
        am4 = selectmodel(GARCH, datat; dist=StdTDist, meanspec=NoIntercept, show_trace=true)
        am5 = selectmodel(GARCH, datam; dist=StdTDist, show_trace=true)
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
    @testset "Standardized" begin
        using Distributions
        MyStdTDist=Standardized{TDist}
        ARCH.startingvals(::Type{<:MyStdTDist}, data::Vector{T}) where T = T[3.]
        Random.seed!(1)
        am = simulate(GARCH{1, 1}([1, 0.9, .05]), 1000, dist=MyStdTDist(3.))
        @test  loglikelihood(fit(am)) ≈ -2700.9089012063323
    end
end
=#
