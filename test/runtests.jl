using ARCH
using Base.Test

coefs = (1., .9, .05)
T = GARCH{1, 1}
srand(1)
data = simulate(T, 10^4, coefs)
ht = zeros(data)
m = ARCHModel(T, data, coefs)
LL = ARCH.arch_loglik!(T, data, ht, coefs...)
@test LL == -29179.73403413289
@test loglikelihood(m) == LL
coefs32 = NTuple{3,Float32}(coefs)
srand(1)
data32 = simulate(T, 10^4, coefs32)
ht32 = zeros(data32)
m32 = ARCHModel(T, data32, coefs32)
LL32 = ARCH.arch_loglik!(T, data32, ht32, coefs32...)
@test LL ≈ LL32
@test loglikelihood(m32) == LL32

@test all(fit(GARCH, data, 3, 3).coefs .≈ (0.9339850882732338, 0.9030897598910242, 0.03901849907497808, 0.019404554001798847, -0.00684168051842598))
