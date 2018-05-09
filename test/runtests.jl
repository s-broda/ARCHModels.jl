using Base.Test

using ARCH
T = 10^4;
spec = GARCH{1, 1};
coefs = [1., .9, .05];
srand(1);
data = simulate(spec, T, coefs);
ht = zeros(data);
am = selectmodel(GARCH, data)
am2 = ARCHModel(spec, data, ht, coefs)
fit!(am2)
am3 = fit(am2)


@test loglikelihood(ARCHModel(spec, data, coefs)) ==  ARCH.loglik!(ht, spec, data, coefs)
@test nobs(am) == T
@test dof(am) == 3
@test coefnames(GARCH{1, 1}) == ["ω", "β₁", "α₁"]
@test all(isapprox.(coef(am), [0.9086850084210619, 0.9055267307122488, 0.050365843108442374], rtol=1e-4))
@test all(isapprox.(stderr(am), [0.14583357347889914, 0.01035533071207874, 0.005222909457230848], rtol=1e-4))
@test all(am2.coefs .== am.coefs)
@test all(am3.coefs .== am2.coefs)

e = @test_throws ARCH.NumParamError ARCH.loglik!(ht, spec, data, [0., 0., 0., 0.])
str = sprint(showerror, e.value)
@test startswith(str, "incorrect number of parameters")
@test_throws ARCH.NumParamError ARCH.sim!(ht, spec, data, [0., 0., 0., 0.])
e = @test_throws ARCH.LengthMismatchError ARCHModel(spec, data, coefs, coefs)
str = sprint(showerror, e.value)
@test startswith(str, "length of arrays does not match")
@test selectmodel(ARCH._ARCH, data).coefs == fit(ARCH._ARCH{3}, data).coefs
io = IOBuffer()
str = sprint(io -> show(io, am))
@test startswith(str, "\nGARCH{1,1}")

d = StdNormal()
@test ARCH.constraints(StdNormal, Float64) == (Float64[], Float64[])
srand(1)
data = rand(d, 10000)
@test ARCH.logkernel(StdNormal, data[1], []) ≈ -0.044190072874578434
@test ARCH.nparams(StdNormal) == 0
@test ARCH.logconst(StdNormal, []) ≈ -0.9189385332046728

d = StdTDist(4)
srand(1)
data = rand(d, 10000)
@test fit(StdTDist, data).ν ≈ 3.9724379269755077
