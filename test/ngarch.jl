@testset "NGARCH" begin
    @test ARCHModels.nparams(NGARCH{1, 1}) == 4
    @test ARCHModels.nparams(NGARCH{2, 3}) == 7
    @test ARCHModels.presample(NGARCH{1, 2}) == 2
    @test ARCHModels.presample(NGARCH{2, 1}) == 2
    spec = NGARCH{1,1}([1., .8, .05, 1.5]);
    str = sprint(show, "text/plain", spec)
    if VERSION < v"1.5.5"
        @test startswith(str, "NGARCH{1,1} specification.\n\n────────────────────────────────\n               ω   β₁    α₁    δ\n────────────────────────────────\nParameters:  1.0  0.8  0.05  1.5\n────────────────────────────────")
    else
        @test startswith(str, "NGARCH{1, 1} specification.\n\n────────────────────────────────\n               ω   β₁    α₁    δ\n────────────────────────────────\nParameters:  1.0  0.8  0.05  1.5\n────────────────────────────────")
    end
    am = simulate(NGARCH{1, 1}([0.05, 0.8, 0.1, 1.5]), T; meanspec=Intercept(3), rng=StableRNG(1))
    am7 = selectmodel(NGARCH, am.data; maxlags=2, show_trace=true)
    @test all(isapprox(coef(am7), [0.051245008765730626,
                                   0.7720495769425677,
                                   0.10455043069060078,
                                   1.7164136264252792,
                                   3.0018869174934437], rtol=1e-4))
    @test coefnames(NGARCH{1, 1}) == ["ω", "β₁", "α₁", "δ"]
    @test coefnames(NGARCH{2, 2}) == ["ω", "β₁", "β₂", "α₁", "α₂", "δ"]
    @test predict(am7) ≈ 0.5152168553581662
    @test predict(am7, :variance) ≈ 0.26544840804515757
    @test predict(am7, :return) ≈ 3.0018869174934437
    @test predict(am7, :VaR) ≈ -1.8033132813609638
    @test predict.(am7, :variance, 1:3) == [predict(am7, :variance, h) for h in 1:3]
    @test ARCHModels.nparams(NGARCH{2, 2}, (1, 1)) == 4

    mask = ARCHModels.subsetmask(NGARCH{2, 2}, (1, 1))
    @test mask == [true, true, false, true, false, true]
    @test ARCHModels.subsettuple(NGARCH{2, 2}, mask) == (1, 1)

    @test_throws ARCHModels.NumParamError NGARCH{1, 1}([1.])

    # Higgins-Bera / HL2005: NGARCH nests GARCH when δ=2.
    g = fit(GARCH{1, 1}, BG96; meanspec=NoIntercept)
    amnest = UnivariateARCHModel(NGARCH{1, 1}(vcat(g.spec.coefs, 2.0)), BG96; meanspec=NoIntercept())
    @test loglikelihood(amnest) ≈ loglikelihood(g) rtol=1e-8
    @test volatilities(amnest) ≈ volatilities(g) rtol=1e-8
    @test ARCHModels.uncond(NGARCH{1, 1}, [1., .9, .05, 2.0]) ≈ ARCHModels.uncond(GARCH{1, 1}, [1., .9, .05])

    # Simulated from GARCH: fitted NGARCH δ is close to 2 and GARCH lags match.
    gsim = simulate(GARCH{1, 1}([1., .9, .05]), T; rng=StableRNG(1), meanspec=NoIntercept())
    ngfit = fit(NGARCH{1, 1}, gsim.data; meanspec=NoIntercept())
    gfit = fit(GARCH{1, 1}, gsim.data; meanspec=NoIntercept())
    @test coef(ngfit)[end] ≈ 2.0 atol=0.5
    @test coef(ngfit)[2] ≈ coef(gfit)[2] rtol=0.05
    @test coef(ngfit)[3] ≈ coef(gfit)[3] atol=0.02

    sm = selectmodel(NGARCH, am.data; meanspec=NoIntercept(), maxlags=2)
    @test sm.spec isa NGARCH

    for dist in (StdNormal(), StdT(5.), StdSkewT(5., -0.2), StdGED(1.5))
        amd = simulate(NGARCH{1, 1}([0.05, 0.8, 0.1, 1.5]), 1500; dist=dist, rng=StableRNG(1), meanspec=NoIntercept())
        fit!(amd)
        @test isfitted(amd)
        @test all(isfinite.(coef(amd)))
    end
end
