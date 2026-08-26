### NGARCH
The NGARCH{p, q} (Nonlinear GARCH) specification, due to [Higgins and Bera (1992)](https://doi.org/10.2307/2526988), is APARCH without leverage in the [Hansen and Lunde (2005)](https://doi.org/10.1002/jae.800) nesting:

```math
\sigma_t^\delta=\omega+\sum_{i=1}^q\alpha_i\lvert a_{t-i}\rvert^\delta+\sum_{j=1}^p\beta_j\sigma_{t-j}^\delta,
\quad \omega,\alpha_i,\beta_j,\delta>0.
```
Coefficients are stored as ``(\omega,\beta_1,\ldots,\beta_p,\alpha_1,\ldots,\alpha_q,\delta)``, so `nparams = p+q+2`. A typical NGARCH(1,1) is `NGARCH{1, 1}`. With ``\delta=2`` this nests [`GARCH{p, q}`](@ref). It is a standalone specification rather than an alias for `APARCH{0}` (APARCH is not on master yet). The corresponding type is [`NGARCH{p, q}`](@ref):
```jldoctest
julia> NGARCH{1, 1}([1., .8, .05, 1.5])
NGARCH{1, 1} specification.

────────────────────────────────
               ω   β₁    α₁    δ
────────────────────────────────
Parameters:  1.0  0.8  0.05  1.5
────────────────────────────────
```
