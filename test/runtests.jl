using Base.Test

using ARCH
T = 10^4;
spec = GARCH{1, 1};
coefs = [1., .9, .05];
srand(1);
data = simulate(spec, T, coefs);
srand(1);
datat = simulate(spec, T, coefs, StdTDist(4))
ht = zeros(data);
am = selectmodel(GARCH, data)
am2 = ARCHModel(spec, data, ht, coefs)
fit!(am2)
am3 = fit(am2)
am4 = selectmodel(GARCH, datat, StdTDist)
am5 = fit(GARCH{3, 0}, data)
@test loglikelihood(ARCHModel(spec, data, coefs)) ==  ARCH.loglik!(ht, spec, StdNormal{}, data, coefs)
@test nobs(am) == T
@test dof(am) == 3

@test coefnames(GARCH{1, 1}) == ["ω", "β₁", "α₁"]
@test all(isapprox.(coef(am), [0.9086850084210619, 0.9055267307122488, 0.050365843108442374], rtol=1e-4))
@test all(isapprox.(stderr(am), [0.14583357347889914, 0.01035533071207874, 0.005222909457230848], rtol=1e-4))
@test all(am2.coefs .== am.coefs)
@test all(am3.coefs .== am2.coefs)
@test all(isapprox(coef(am4), [0.8306902920885605, 0.9189514425541352, 0.04207946140844637, 3.8356660627658075], rtol=1e-4))

@test_warn "inaccurate" stderr(am5)
@test_warn "inaccurate" stderr(ARCHModel(GARCH{3, 0}, data, [1., .1, .2, .3]))
e = @test_throws ARCH.NumParamError ARCH.loglik!(ht, spec, StdNormal{}, data, [0., 0., 0., 0.])
str = sprint(showerror, e.value)
@test startswith(str, "incorrect number of parameters")
@test_throws ARCH.NumParamError ARCH.sim!(ht, spec, StdNormal(), data, [0., 0., 0., 0.])
e = @test_throws ARCH.LengthMismatchError ARCHModel(spec, data, coefs, coefs)
str = sprint(showerror, e.value)
@test startswith(str, "length of arrays does not match")
io = IOBuffer()
str = sprint(io -> show(io, am))
@test startswith(str, "\nGARCH{1,1}")

@test selectmodel(ARCH._ARCH, data).coefs == fit(ARCH._ARCH{3}, data).coefs


@test fit(StdNormal, data).coefs == Float64[]
@test coefnames(StdNormal) == String[]
@test ARCH.distname(StdNormal) == "Gaussian"

srand(1)
data = rand(StdTDist(4), 10000)
@test fit(StdTDist, data).coefs[1] ≈ 3.972437329588246 rtol=1e-4
@test coefnames(StdTDist) == ["ν"]
@test ARCH.distname(StdTDist) == "Student's t"
