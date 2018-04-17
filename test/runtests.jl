using ARCH
using Base.Test

T=10^4
spec = GARCH{1, 1}
coefs = (1., .9, .05)
srand(1)
data = simulate(spec, T, coefs)
ht = zeros(data)
coefs32 = NTuple{3,Float32}(coefs)
srand(1)
data32 = simulate(spec, T, coefs32)
ht32 = zeros(data32)

AM=selectmodel(GARCH, data)
@test loglikelihood(ARCHModel(spec, data, coefs)) ==  ARCH.arch_loglik!(spec, data, ht, [coefs...])
@test nobs(AM) == T
@test dof(AM) == 3
println(AM.coefs)
@test all(AM.coefs .≈ (0.9086851798550601, 0.9055267194855281, 0.05036584604511605))

AM32=selectmodel(GARCH, data32)
@test loglikelihood(ARCHModel(spec, data32, coefs32)) ==  ARCH.arch_loglik!(spec, data32, ht32, [coefs32...])
@test nobs(AM32) == T
@test dof(AM32) == 3
println(AM32.coefs)
@test all(AM32.coefs .≈ (1.0329274f0, 0.89497274f0, 0.055076957f0))
