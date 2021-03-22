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
TGARCH{0,0,2} specification.

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
TGARCH{0,1,1} specification.

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
TGARCH{1,1,1} specification.

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
EGARCH{1,1,1} specification.

─────────────────────────────────
                ω   γ₁   β₁    α₁
─────────────────────────────────
Parameters:  -0.1  0.1  0.9  0.04
─────────────────────────────────
```
## [Mean specifications](@id meanspec)
Mean specifications serve to specify ``\mu_t``. They are modelled as subtypes of [`MeanSpec`](@ref). They contain their parameters as (possibly empty) vectors, but convenience constructors are provided where appropriate. The following specifications are available:
* A zero mean: ``\mu_t=0``. Available as [`NoIntercept`](@ref):
```jldoctest TYPES
julia> NoIntercept() # convenience constructor, eltype defaults to Float64
NoIntercept{Float64}(Float64[])
```
* An intercept: ``\mu_t=\mu``. Available as [`Intercept`](@ref):
```jldoctest TYPES
julia> Intercept(3) # convenience constructor
Intercept{Float64}([3.0])
```

* A linear regression model: ``\mu_t=\mathbf{x}_t^{\mathrm{\scriptscriptstyle T}}\boldsymbol{\beta}``. Available as [`Regression`](@ref):
```jldoctest TYPES
julia> X = ones(100, 1);

julia> reg = Regression(X);
```
In this example, we created a regression model containing one regressor, given by a column of ones; this is equivalent to including an intercept in the model (see [`Intercept`](@ref) above). In general, the constructor should be passed a design matrix ``\mathbf{X}`` containing ``\{\mathbf{x}_t^{\mathrm{\scriptscriptstyle T}}\}_{t=1\ldots T}`` as its rows; that is, for a model with ``T`` observations and ``k`` regressors, ``X`` would have dimensions ``T\times k``.

Another way to create a linear regression with ARCH errors is to pass a `LinearModel` or `DataFrameRegressionModel` from [GLM.jl](https://github.com/JuliaStats/GLM.jl) to [`fit`](@ref), as described under [Integration with GLM.jl](@ref).

* An ARMA(p, q) model: ``\mu_t=c+\sum_{i=1}^p \varphi_i r_{t-i}+\sum_{i=1}^q \theta_i a_{t-i}``. Available as [`ARMA{p, q}`](@ref):
```jldoctest TYPES
julia> ARMA{1, 1}([1., .9, -.1])
ARMA{1,1,Float64}([1.0, 0.9, -0.1])
```
Pure AR(p) and MA(q) models are obtained as follows:
```jldoctest TYPES
julia> AR{1}([1., .9])
ARMA{1,0,Float64}([1.0, 0.9])
julia> MA{1}([1., -.1])
ARMA{0,1,Float64}([1.0, -0.1])
```

## Distributions
### Built-in distributions
Different standardized (mean zero, variance one) distributions for ``z_t`` are available as subtypes of [`StandardizedDistribution`](@ref). `StandardizedDistribution` in turn subtypes `Distribution{Univariate, Continuous}` from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl), though not the entire interface need necessarily be implemented. `StandardizedDistribution`s again hold their parameters as vectors, but convenience constructors are provided. The following are currently available:
* [`StdNormal`](@ref), the standard [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution):
```jldoctest TYPES
julia> StdNormal() # convenience constructor
StdNormal{Float64}(coefs=Float64[])
```
* [`StdT`](@ref), the standardized [Student's ``t`` distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution):
```jldoctest TYPES
julia> StdT(3) # convenience constructor
StdT{Float64}(coefs=[3.0])
```
* [`StdSkewT`](@ref), the standardized [Hansen skewed ``t`` distribution](https://en.wikipedia.org/wiki/Skewed_generalized_t_distribution#cite_note-hansen-8):
```jldoctest TYPES
julia> StdSkewT(3, -0.3) # convenience constructor
StdSkewT{Float64}(coefs=[3.0, -0.3])
```
* [`StdGED`](@ref), the standardized [Generalized Error Distribution](https://en.wikipedia.org/wiki/Generalized_normal_distribution):
```jldoctest TYPES
julia> StdGED(1) # convenience constructor
StdGED{Float64}(coefs=[1.0])
```
### User-defined standardized distributions
Apart from the natively supported standardized distributions, it is possible to wrap a continuous univariate distribution from the [Distributions package](https://github.com/JuliaStats/Distributions.jl) in the [`Standardized`](@ref) wrapper type. Below, we reimplement the standardized normal distribution:

```jldoctest TYPES
julia> using Distributions

julia> const MyStdNormal = Standardized{Normal};
```

`MyStdNormal` can be used whereever a built-in distribution could, albeit with a speed penalty. Note also that if the underlying distribution (such as `Normal` in the example above) contains location and/or scale parameters, then these are no longer identifiable, which implies that the estimated covariance matrix of the estimators will be singular.

A final remark concerns the domain of the parameters: the estimation process relies on a starting value for the parameters of the distribution, say ``\theta\equiv(\theta_1, \ldots, \theta_p)'``. For a distribution wrapped in [`Standardized`](@ref), the starting value for ``\theta_i`` is taken to be a small positive value ϵ. This will fail if ϵ is not in the domain of ``\theta_i``; as an example, the standardized Student's ``t`` distribution is only defined for degrees of freedom larger than 2, because a finite variance is required for standardization. In that case, it is necessary to define a method of the (non-exported) function `startingvals` that returns a feasible vector of starting values, as follows:

```jldoctest TYPES
julia> const MyStdT = Standardized{TDist};

julia> ARCHModels.startingvals(::Type{<:MyStdT}, data::Vector{T}) where T = T[3.]
```
## Working with UnivariateARCHModels
The constructor for [`UnivariateARCHModel`](@ref) takes two mandatory arguments: an instance of a subtype of [`UnivariateVolatilitySpec`](@ref), and a vector of returns. The mean specification and error distribution can be changed via the keyword arguments `meanspec` and `dist`, which respectively default to `NoIntercept` and `StdNormal`.

For example, to construct a GARCH(1, 1) model with an intercept and ``t``-distributed errors, one would do
```jldoctest TYPES
julia> spec = GARCH{1, 1}([1., .9, .05]);

julia> data = BG96;

julia> am = UnivariateARCHModel(spec, data; dist=StdT(3.), meanspec=Intercept(1.))

TGARCH{0,1,1} model with Student's t errors, T=1974.


──────────────────────────────
                             μ
──────────────────────────────
Mean equation parameters:  1.0
──────────────────────────────
─────────────────────────────────────────
                             ω   β₁    α₁
─────────────────────────────────────────
Volatility parameters:     1.0  0.9  0.05
─────────────────────────────────────────
──────────────────────────────
                             ν
──────────────────────────────
Distribution parameters:   3.0
──────────────────────────────
```

The model can then be fitted as follows:

```jldoctest TYPES
julia> fit!(am)

TGARCH{0,1,1} model with Student's t errors, T=1974.

Mean equation parameters:
─────────────────────────────────────────────
     Estimate   Std.Error   z value  Pr(>|z|)
─────────────────────────────────────────────
μ  0.00227251  0.00686797  0.330885    0.7407
─────────────────────────────────────────────

Volatility parameters:
──────────────────────────────────────────────
      Estimate   Std.Error   z value  Pr(>|z|)
──────────────────────────────────────────────
ω   0.00232225  0.00163588   1.41958    0.1557
β₁  0.884488    0.0369039   23.9673     <1e-99
α₁  0.124866    0.0404843    3.08429    0.0020
──────────────────────────────────────────────

Distribution parameters:
─────────────────────────────────────────
   Estimate  Std.Error  z value  Pr(>|z|)
─────────────────────────────────────────
ν   4.11211   0.400396  10.2701    <1e-24
─────────────────────────────────────────
```

It should, however, rarely be necessary to construct a `UnivariateARCHModel` manually via its constructor; typically, instances of it are created by calling [`fit`](@ref), [`selectmodel`](@ref), or [`simulate`](@ref).

!!! note
    If you *do* manually construct a `UnivariateARCHModel`, be aware that the constructor does not create copies of its arguments. This means that, e.g., calling `simulate!` on the constructed model will modify your data vector:
    ```jldoctest TYPES
    julia> mydata = copy(BG96); mydata[end]
    0.528047

    julia> am = UnivariateARCHModel(ARCH{0}([1.]), mydata);

    julia> simulate!(am);

    julia> mydata[end] ≈ 0.528047
    false
    ```
As discussed earlier, [`UnivariateARCHModel`](@ref) implements the interface of `StatisticalModel` from [`StatsBase`](http://juliastats.github.io/StatsBase.jl/stable/statmodels.html), so you
can call `coef`, `coefnames`, `confint`, `dof`, `informationmatrix`, `isfitted`, `loglikelihood`, `nobs`,  `score`, `stderror`, `vcov`, etc. on its instances:

```jldoctest TYPES
julia> nobs(am)
1974
```

Other useful methods include [`means`](@ref), [`volatilities`](@ref) and [`residuals`](@ref).
