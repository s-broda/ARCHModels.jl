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
