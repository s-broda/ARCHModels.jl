using ARCH
using Base.Test

T=10^4
spec = GARCH{1, 1}

coefs = (1., .9, .05)
srand(1)
data = simulate(spec, T, coefs)
ht = zeros(data)
@test loglikelihood(ARCHModel(spec, data, coefs)) ==  ARCH.arch_loglik!(spec, data, ht, coefs...)
AM=fit(GARCH, data, 3, 3)
@test all(AM.coefs .≈ (0.9339850882732338, 0.9030897598910242, 0.03901849907497808, 0.019404554001798847, -0.00684168051842598))
@test nobs(AM) == T
@test dof(AM) == 5

coefs32 = NTuple{3,Float32}(coefs)
srand(1)
data32 = simulate(spec, T, coefs32)
ht32 = zeros(data32)
@test loglikelihood(ARCHModel(spec, data32, coefs32)) ==  ARCH.arch_loglik!(spec, data32, ht32, coefs32...)
AM32=fit(GARCH, data32, 3, 3)
@test all(AM32.coefs .≈ (1.0410453f0, 0.8948746f0, 0.03434224f0, 0.027093302f0, -0.0065628835f0))
@test nobs(AM32) == T
@test dof(AM32) == 5
