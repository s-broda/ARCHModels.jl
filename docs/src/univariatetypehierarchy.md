# Univariate
An instance of [`UnivariateARCHModel`](@ref) contains a vector of data (such as equity returns), and encapsulates information about the [volatility specification](@ref volaspec) (e.g., [GARCH](@ref) or [EGARCH](@ref)), the [mean specification](@ref meanspec) (e.g., whether an intercept is included), and the [error distribution](@ref Distributions). 

In general a univariate model can be written
```math
r_t = \mu_t + \sigma_t z_t, \quad z_t \stackrel{\text{iid}}{\sim} F.
```
Hence, a univariate model is a triple of functions ``\left(\mu_t, \sigma_t, F \right)``.
The table below lists current options for the conditional mean, conditional variance, and the error distribution.


| ``\mu_t`` 	| ``\sigma_t`` 	| ``F`` 	|
| --- | --- | --- |
| `NoIntercept` 	| `ARCH{0}` (constant) 	| `StdNormal` 	|
| `Intercept` 	| `ARCH{q}` 	| `StdT` 	|
| `ARMA{p,q}` 	| `GARCH{p,q}` 	| `StdGED` 	|
| `Regression(X)` 	| `TGARCH{o,p,q}` 	| Std User-Defined 	|
|  	| `EGARCH{o,p,q}` 	|  	|
|  	| `NGARCH{p,q}` 	|  	|

Details on these options are given below.
## [Volatility specifications](@id volaspec)
Volatility specifications describe the evolution of ``\sigma_t``. They are modelled as subtypes of [`UnivariateVolatilitySpec`](@ref). There is one type for each class of (G)ARCH model, parameterized by the number(s) of lags (e.g., ``p``, ``q`` for a GARCH(p, q) model). For each volatility specification, the order of the parameters in the coefficient vector is such that all parameters pertaining to the first type parameter (``p``) appear before those pertaining to the second (``q``).
### ARCH
With ``a_t\equiv r_t-\mu_t``, the ARCH(q) volatility specification, due to [Engle (1982)](https://doi.org/10.2307/1912773 ), is
```math
\sigma_t^2=\omega+\sum_{i=1}^q\alpha_i a_{t-i}^2, \quad \omega, \alpha_i>0,\quad \sum_{i=1}^{q} \alpha_i<1.
```
The corresponding type is [`ARCH{q}`](@ref). For example, an ARCH(2) model with ``ω=1``, ``α₁=.5``, and ``α₂=.4`` is obtained with
```jldoctest TYPES
julia> using ARCHModels

julia> ARCH{2}([1., .5, .4])
TGARCH{0, 0, 2} specification.

──────────────────────────
               ω   α₁   α₂
──────────────────────────
Parameters:  1.0  0.5  0.4
──────────────────────────
```

### GARCH
The GARCH(p, q) model, due to [Bollerslev (1986)](https://doi.org/10.1016/0304-4076(86)90063-1), specifies the volatility as
```math
\sigma_t^2=\omega+\sum_{i=1}^p\beta_i \sigma_{t-i}^2+\sum_{i=1}^q\alpha_i a_{t-i}^2, \quad \omega, \alpha_i, \beta_i>0,\quad \sum_{i=1}^{\max p,q} \alpha_i+\beta_i<1.
```
It is available as [`GARCH{p, q}`](@ref):
```jldoctest TYPES
julia> GARCH{1, 1}([1., .9, .05])
GARCH{1, 1} specification.

───────────────────────────
               ω   β₁    α₁
───────────────────────────
Parameters:  1.0  0.9  0.05
───────────────────────────
```
This creates a GARCH(1, 1) specification with ``ω=1``, ``β=.9``, and ``α=.05``.

### TGARCH
As may have been guessed from the output above, the ARCH and GARCH models are actually special cases of a more general class of models, known as TGARCH (Threshold GARCH), due to [Glosten, Jagannathan, and Runkle (1993)](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x). The TGARCH{o, p, q} model takes the form

```math
\sigma_t^2=\omega+\sum_{i=1}^o\gamma_i  a_{t-i}^2 1_{a_{t-i}<0}+\sum_{i=1}^p\beta_i \sigma_{t-i}^2+\sum_{i=1}^q\alpha_i a_{t-i}^2, \quad \omega, \alpha_i, \beta_i, \gamma_i>0, \sum_{i=1}^{\max o,p,q} \alpha_i+\beta_i+\gamma_i/2<1.
```

The TGARCH model allows the volatility to react differently (typically more strongly) to negative shocks, a feature known as the (statistical) leverage effect. Is available as [`TGARCH{o, p, q}`](@ref):

```jldoctest TYPES
julia> TGARCH{1, 1, 1}([1., .04, .9, .01])
TGARCH{1, 1, 1} specification.

─────────────────────────────────
               ω    γ₁   β₁    α₁
─────────────────────────────────
Parameters:  1.0  0.04  0.9  0.01
─────────────────────────────────
```

### EGARCH
The EGARCH{o, p, q} volatility specification, due to [Nelson (1991)](https://doi.org/10.2307/2938260), is
```math
\log(\sigma_t^2)=\omega+\sum_{i=1}^o\gamma_i z_{t-i}+\sum_{i=1}^p\beta_i \log(\sigma_{t-i}^2)+\sum_{i=1}^q\alpha_i (|z_{t-i}|-\sqrt{2/\pi}), \quad z_t=r_t/\sigma_t,\quad \sum_{i=1}^{p}\beta_i<1.
```

Like the TGARCH model, it can account for the leverage effect. The corresponding type is [`EGARCH{o, p, q}`](@ref):
```jldoctest TYPES
julia> EGARCH{1, 1, 1}([-0.1, .1, .9, .04])
EGARCH{1, 1, 1} specification.

─────────────────────────────────
                ω   γ₁   β₁    α₁
─────────────────────────────────
Parameters:  -0.1  0.1  0.9  0.04
─────────────────────────────────
```

### NGARCH
The NGARCH{p, q} (Nonlinear GARCH) specification, due to [Higgins and Bera (1992)](https://doi.org/10.2307/2526988), is APARCH without leverage in the [Hansen and Lunde (2005)](https://doi.org/10.1002/jae.800) nesting:

```math
\sigma_t^\delta=\omega+\sum_{i=1}^q\alpha_i\lvert a_{t-i}\rvert^\delta+\sum_{j=1}^p\beta_j\sigma_{t-j}^\delta,
\quad \omega,\alpha_i,\beta_j,\delta>0.
```
Coefficients are stored as ``(\omega,\beta_1,\ldots,\beta_p,\alpha_1,\ldots,\alpha_q,\delta)``, so `nparams = p+q+2`. A typical NGARCH(1,1) is `NGARCH{1, 1}`. With ``\delta=2`` this nests [`GARCH{p, q}`](@ref). It is a standalone specification rather than an alias for `APARCH{0}` (APARCH is not on master yet). The corresponding type is [`NGARCH{p, q}`](@ref):
```jldoctest TYPES
julia> NGARCH{1, 1}([1., .8, .05, 1.5])
NGARCH{1, 1} specification.

────────────────────────────────
               ω   β₁    α₁    δ
────────────────────────────────
Parameters:  1.0  0.8  0.05  1.5
────────────────────────────────
```
