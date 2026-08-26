    @test ARCHModels.nparams(APARCH{1, 1, 1}, ()) == 2

    lower, upper = ARCHModels.constraints(APARCH{1, 1, 1}, Float64)
    @test lower[1] == 0 && lower[3] == 0 && lower[4] == 0
    @test lower[2] == -1 && upper[2] == 1
    @test upper[1] == Inf && upper[end] == Inf

    sv = ARCHModels.startingvals(APARCH{1, 1, 1}, BG96)
    @test length(sv) == 5
    @test sv[2] ≈ 0.1
    @test sv[end] == 2
    @test sv[1] > 0 && isfinite(sv[1])

    svs = ARCHModels.startingvals(APARCH{2, 2, 2}, BG96, (0, 1, 1))
    @test length(svs) == 8
    @test svs[2] == 0 && svs[3] == 0 && svs[5] == 0 && svs[7] == 0
    @test svs[end] == 2
    @test svs[1] > 0 && isfinite(svs[1])
    mask2 = ARCHModels.subsetmask(APARCH{2, 2, 2}, (1, 1))
    @test mask2 == ARCHModels.subsetmask(APARCH{2, 2, 2}, (0, 1, 1))
    @test ARCHModels.subsettuple(APARCH{2, 2, 2}, mask2) == (0, 1, 1)
    @test ARCHModels.startingvals(APARCH{2, 2, 2}, BG96, (1, 1)) == svs

    # uncond with leverage γ≠0, δ≠2, and q>o (γi=0 for extra ARCH lags)
    u_lev = ARCHModels.uncond(APARCH{1, 1, 2}, [1., 0.2, 0.3, 0.1, 0.05, 1.4])
    @test u_lev > 0 && isfinite(u_lev)
    @test u_lev ≈ ARCHModels.uncond(APARCH{2, 1, 2}, [1., 0.2, 0.0, 0.3, 0.1, 0.05, 1.4])
    @test u_lev != ARCHModels.uncond(APARCH{1, 1, 2}, [1., 0.0, 0.3, 0.1, 0.05, 1.4])
    # σδ <= 0 when the process is nonstationary (not the GARCH nesting case)
    @test ARCHModels.uncond(APARCH{0, 1, 1}, [1., 0.6, 0.6, 2.0]) <= 0

    ht = [1.0, 1.2]
    lht = [0.0, log(1.2)]
    zt = [0.5, -0.3]
    at = [0.5, -0.4]
    coefs111 = [0.05, 0.1, 0.8, 0.1, 1.5]
    ARCHModels.update!(ht, lht, zt, at, APARCH{1, 1, 1}, coefs111, 1)
    push!(zt, 0.)
    push!(at, 0.)
    nht = length(ht)
    ARCHModels.update!(ht, lht, zt, at, APARCH{1, 1, 1}, coefs111, 2)
    @test length(ht) == nht + 1
    @test isfinite(ht[end])

    htq = [1.0, 1.1]
    lhtq = log.(htq)
    ztq = [0.3, -0.4]
    atq = [0.3, -0.5]
    ARCHModels.update!(htq, lhtq, ztq, atq, APARCH{1, 1, 2}, [0.05, 0.1, 0.7, 0.08, 0.05, 1.5], 1)
    @test isfinite(htq[end])
    amq = simulate(APARCH{1, 1, 2}([0.05, 0.1, 0.7, 0.08, 0.05, 1.5]), 500; rng=StableRNG(1), meanspec=NoIntercept())
    @test nobs(amq) == 500
    @test all(isfinite.(volatilities(amq)))

    ht0 = [1.0]
    lht0 = [0.0]
    zt0 = [0.0]
    at0 = [1.0]
    ARCHModels.update!(ht0, lht0, zt0, at0, APARCH{1, 1, 1}, [0., 0., 0., 0., 2.], 1)
    @test ht0[end] == 0
    @test lht0[end] == -ht0[end]

    am222 = simulate(APARCH{2, 2, 2}([0.05, 0.1, 0.05, 0.4, 0.3, 0.08, 0.04, 1.5]), 500; rng=StableRNG(1), meanspec=NoIntercept())
    @test nobs(am222) == 500
    @test all(isfinite.(volatilities(am222)))

    amfit = fit(APARCH{1, 1, 1}, BG96; meanspec=NoIntercept)
    @test isfitted(amfit)
    @test all(isfinite.(coef(amfit)))
