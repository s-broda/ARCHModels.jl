using Test

@testset "IGARCH" begin
    @test ARCHModels.nparams(IGARCH{1, 2}) == 3
    @test ARCHModels.nparams(IGARCH{2, 1}) == 3
    @test ARCHModels.presample(IGARCH{1, 2}) == 2
    spec = IGARCH{1,1}([.1, .05]);
    str = sprint(show, "text/plain", spec)
    if VERSION < v"1.5.5"
        @test startswith(str, "IGARCH{1,1} specification.\n\n────────────────────────────────────\n               ω    α₁  β₁ (implied)\n────────────────────────────────────\nParameters:  0.1  0.05          0.95\n────────────────────────────────────")
    else
        @test startswith(str, "IGARCH{1, 1} specification.\n\n────────────────────────────────────\n               ω    α₁  β₁ (implied)\n────────────────────────────────────\nParameters:  0.1  0.05          0.95\n────────────────────────────────────")
    end
    am = simulate(IGARCH{1, 1}([0.05, 0.05]), T; meanspec=NoIntercept(), rng=StableRNG(1))
    am7 = selectmodel(IGARCH, am.data; meanspec=NoIntercept(), maxlags=2, show_trace=true)
    @test all(isapprox(coef(am7), [0.055500078224673344,
                                   0.052571777858308424], rtol=1e-4))
    @test coefnames(IGARCH{2, 2}) == ["ω", "β₁", "α₁", "α₂"]
    @test coefnames(IGARCH{1, 1}) == ["ω", "α₁"]
    @test predict(am7) ≈ 7.543765698147407
    @test predict(am7, :variance) ≈ 56.90840090854544
    @test predict(am7, :return) ≈ 0.0
    @test predict(am7, :VaR) ≈ 17.54942329414748
    # multi-step variance forecasts rise linearly in ω, not mean-revert
    h1 = predict(am7, :variance, 1)
    h2 = predict(am7, :variance, 2)
    h3 = predict(am7, :variance, 3)
    @test h2 ≈ h1 + am7.spec.coefs[1]
    @test h3 ≈ h1 + 2*am7.spec.coefs[1]
    @test predict.(am7, :variance, 1:3) == [predict(am7, :variance, h) for h in 1:3]
    @test ARCHModels.uncond(IGARCH{1, 1}, [0.05, 0.05]) == Inf
    @test ARCHModels.implied_beta(am7.spec) ≈ 1 - am7.spec.coefs[2]
    @test ARCHModels.nparams(IGARCH{2, 2}, (1, 1)) == 2

    mask = ARCHModels.subsetmask(IGARCH{3, 3}, (1, 1))
    @test mask == [true, false, false, true, false, false]
    @test ARCHModels.subsettuple(IGARCH{3, 3}, mask) == (1, 1)
    mask22 = ARCHModels.subsetmask(IGARCH{3, 3}, (2, 2))
    @test mask22 == [true, true, false, true, true, false]
    @test ARCHModels.subsettuple(IGARCH{3, 3}, mask22) == (2, 2)

    @test_throws ARCHModels.NumParamError IGARCH{1, 1}([1.])
    @test_throws ArgumentError IGARCH{0, 1}([1., .5])

    for dist in (StdNormal(), StdT(5.), StdSkewT(5., -0.2), StdGED(1.5))
        amd = simulate(IGARCH{1, 1}([0.05, 0.05]), 1500; dist=dist, rng=StableRNG(1), meanspec=NoIntercept())
        fit!(amd)
        @test isfitted(amd)
        @test all(isfinite.(coef(amd)))
    end
end
