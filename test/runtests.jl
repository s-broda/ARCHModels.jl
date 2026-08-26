using Test

using ARCHModels
using GLM
using DataFrames
using StableRNGs


T = 10^4;

@testset "TGARCH" begin
    @test ARCHModels.nparams(TGARCH{1, 2, 3}) == 7
    @test ARCHModels.presample(TGARCH{1, 2, 3}) == 3
    spec = TGARCH{1,1,1}([1., .05, .9, .01]);
    str = sprint(show, "text/plain", spec)
    if VERSION < v"1.5.5"
        @test startswith(str, "TGARCH{1,1,1} specification.\n\n─────────────────────────────────\n               ω    γ₁   β₁    α₁\n─────────────────────────────────\nParameters:  1.0  0.05  0.9  0.01\n─────────────────────────────────")
    else
        @test startswith(str, "TGARCH{1, 1, 1} specification.\n\n─────────────────────────────────\n               ω    γ₁   β₁    α₁\n─────────────────────────────────\nParameters:  1.0  0.05  0.9  0.01\n─────────────────────────────────")                
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
include("ngarch.jl")
