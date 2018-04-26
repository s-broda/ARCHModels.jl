using Base.Test

using ARCH
T=10^4;
spec = GARCH{1, 1};
coefs = [1., .9, .05];
srand(1);
data = simulate(spec, T, coefs);
ht = zeros(data);
am=selectmodel(GARCH, data)
am2=ARCHModel(spec, data, ht, coefs)
fit!(am2)
am3=fit(am2)


@test loglikelihood(ARCHModel(spec, data, coefs)) ==  ARCH.loglik!(ht, spec, data, coefs)
@test nobs(am) == T
@test dof(am) == 3
@test coefnames(GARCH{1, 1})==["omega", "beta_1", "alpha_1"]
@test all(isapprox.(coef(am), [0.9086850084210619, 0.9055267307122488, 0.050365843108442374], rtol=1e-4))
@test all(am2.coefs .== am.coefs)
@test all(am3.coefs .== am2.coefs)
@test_throws ARCH.NumParamError ARCH.loglik!(ht, spec, data, [0., 0., 0., 0.])
@test_throws ARCH.NumParamError ARCH.sim!(ht, spec, data, [0., 0., 0., 0.])
@test_throws ARCH.LengthMissmatchError ARCHModel(spec, data, coefs, coefs)
@test selectmodel(ARCH._ARCH, data).coefs == fit(ARCH._ARCH{3}, data).coefs

 f = x-> ARCH.loglik(spec, data, x)
 @test det(ForwardDiff.hessian(f, coefs)) .≈ -7.523213634979238e11
