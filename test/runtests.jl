using ARCH
using Base.Test

T=10^4
spec = GARCH{1, 1}

coefs = (1., .9, .05)
srand(1)
data = simulate(spec, T, coefs)
ht = zeros(data)
@test loglikelihood(ARCHModel(spec, data, coefs)) ==  ARCH.arch_loglik!(spec, data, ht, coefs...)
AM=selectmodel(GARCH, data)
@test nobs(AM) == T
@test dof(AM) == 3
@test all(AM.coefs .≈ (0.9086876876511291, 0.9055259279961678, 0.0503666237393654))

coefs32 = NTuple{3,Float32}(coefs)
srand(1)
data32 = simulate(spec, T, coefs32)
ht32 = zeros(data32)
@test loglikelihood(ARCHModel(spec, data32, coefs32)) ==  ARCH.arch_loglik!(spec, data32, ht32, coefs32...)
AM32=selectmodel(GARCH, data32)
@test nobs(AM32) == T
@test dof(AM32) == 3
@test all(AM32.coefs .≈ (0.97600096f0, 0.8996421f0, 0.05328631f0))
