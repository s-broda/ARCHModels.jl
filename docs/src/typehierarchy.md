# Type hierarchy
## [Univariate volatility specifications](@id volaspec)
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
julia> using ARCHModels

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
As may have been guessed from the output above, the ARCH and GARCH models are actually special cases of a more general class of models, known as TGARCH (Threshold GARCH), due to [Glosten, Jagannathan, and Runkle](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x). The TGARCH{o, p, q} model takes the form

```math
\sigma_t^2=\omega+\sum_{i=1}^o\gamma_i  a_{t-i}^2 1_{a_{t-i}<0}+\sum_{i=1}^p\beta_i \sigma_{t-i}^2+\sum_{i=1}^q\alpha_i a_{t-i}^2, \quad \omega, \alpha_i, \beta_i, \gamma_i>0, \sum_{i=1}^{\max o,p,q} \alpha_i+\beta_i+\gamma_i/2<1.
```

The TGARCH model allows the volatility to react differently (typically more strongly) to negative shocks, a feature known as the (statistical) leverage effect. Is available as [`TGARCH{o, p, q}`](@ref):

```jldoctest TYPES
julia> using ARCHModels

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
## [Multivariate volatility and covariance specifications](@id covspec)
The main challenge in multivariate ARCH modelling is the _curse of dimensionality_: allowing each of the ``(d)(d+1)/2`` elements of ``\Sigma_t`` to depend on the past returns of all ``d`` other assets requires ``O(d^4)`` parameters without imposing additional structure. Multivariate ARCH models differ in which structure they impose.

The dynamics of ``\Sigma_t``  are modelled as subtypes of [`MultivariateVolatilitySpec`](@ref). These may be combined with different mean specifications as in the univariate case, and (in principle) with different specifications for the joint distribution of the standardized returns, although at present, only the multivariate standard normal is supported.
### CCC
The CCC (constant conditional correlation) model of [Bollerslev (1990)](https://doi.org/10.2307/2109358) decomposes
``\Sigma_t`` as
```math
\Sigma_t=D_t R_t D_t,
```
where ``R_t`` is the conditional correlation matrix and ``D_t`` is a diagonal matrix containing the volatilities of the individual assets, which are modelled as univariate ARCH processes. In the constant conditional correlation (CCC) model, ``R_t=R`` is assumed constant. The model is typically, including in this package, estimated in a two-step procedure: first, univariate ARCH models are fitted to the $d$ asset returns, and then ``R`` is estimated as the sample correlation matrix of the standardized residuals.

### DCC
The DCC model of [Engle (2002)](https://doi.org/10.1198/073500102288618487) extends the CCC model by making the ``R_t`` dynamic (hence the name, dynamic conditional correlation model). In particular, for a DCC(p, q) model (with covariance targeting),

```math
R_{ij, t} = \frac{Q_{ij,t}}{\sqrt{Q_{ii,t}Q_{jj,t}}},
```
where
```math
Q_{t} \equiv\bar{Q}(1-\bar\alpha-\bar\beta)+\sum_{i=1}^{p} \beta_iQ_{t-i}+\sum_{i=1}^{q}\alpha_i\epsilon_{t-i}\epsilon_{t-i}^\mathrm{\scriptsize T},
```
``\bar{\alpha}\equiv\sum_{i=1}^q\alpha_i``, ``\bar{\beta}\equiv\sum_{i=1}^q\beta_i``, ``\epsilon_{t}\equiv D_t^{-1}a_t$, $Q_{t}=\mathrm{cov}
(\epsilon_t|F_{t-1})``, and ``\bar{Q}=\mathrm{cov}(\epsilon_{t})``.

Like the CCC model, the DCC model can be estimated in two steps, by first fitting univariate ARCH models to the individual assets and saving the standardized residuals ``\{\epsilon_t\}``, and then estimating the DCC parameters from those. [Engle (2002)](https://doi.org/10.1198/073500102288618487) provides the details and expressions for the standard errors. By default, this package employs an alternative estimator due to [Engle, Ledoit, and Wolf (2019)](https://doi.org/10.1080/07350015.2017.1345683) which is better suited to large-dimensional problems. It achieves this by i) estimating ``\bar{Q}`` with a nonlinear shrinkage estimator instead of the sample covariance of $\epsilon_t$, and ii) estimating the DCC parameters by maximizing the sum of the pairwise log-likelihoods, rather than the joint log-likelihood over all assets, thereby avoiding the inversion of large matrices during the optimization.


## [Mean specifications](@id meanspec)
Mean specifications serve to specify ``\mu_t``. They are modelled as subtypes of [`MeanSpec`](@ref). They contain their parameters as (possibly empty) vectors, but convenience constructors are provided where appropriate. Currently, three specifications are available:
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
Different standardized (mean zero, variance one) distributions for ``z_t`` are available as subtypes of [`StandardizedDistribution`](@ref). `StandardizedDistribution` in turn subtypes `Distribution{Univariate, Continuous}` from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl), though not the entire interface must necessarily be implemented. `StandardizedDistribution`s again hold their parameters as vectors, but convenience constructors are provided. The following are currently available:
* [`StdNormal`](@ref), the standard [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution):
```jldoctest TYPES
julia> StdNormal() # convenience constructor
StdNormal{Float64}(coefs=Float64[])
```
* [`StdT`](@ref), the standardized [Student's `t` distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution):
```jldoctest TYPES
julia> StdT(3) # convenience constructor
StdT{Float64}(coefs=[3.0])
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
## Working with ARCHModels
### Univariate
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
μ  0.00227251  0.00686802  0.330882    0.7407
─────────────────────────────────────────────

Volatility parameters:
──────────────────────────────────────────────
      Estimate   Std.Error   z value  Pr(>|z|)
──────────────────────────────────────────────
ω   0.00232225  0.00163909   1.41679    0.1565
β₁  0.884488    0.036963    23.929      <1e-99
α₁  0.124866    0.0405471    3.07952    0.0021
──────────────────────────────────────────────

Distribution parameters:
─────────────────────────────────────────
   Estimate  Std.Error  z value  Pr(>|z|)
─────────────────────────────────────────
ν   4.11211   0.400384  10.2704    <1e-24
─────────────────────────────────────────
```

It should, however, rarely be necessary to construct an `UnivariateARCHModel` manually via its constructor; typically, instances of it are created by calling [`fit`](@ref), [`selectmodel`](@ref), or [`simulate`](@ref).

As discussed earlier, [`UnivariateARCHModel`](@ref) implements the interface of StatisticalModel from [`StatsBase`](http://juliastats.github.io/StatsBase.jl/stable/statmodels.html), so you
can call `coef`, `coefnames`, `confint`, `dof`, `informationmatrix`, `isfitted`, `loglikelihood`, `nobs`,  `score`, `stderror`, `vcov`, etc. on its instances:

```jldoctest TYPES
julia> nobs(am)
1974
```

Other useful methods include [`means`](@ref), [`volatilities`](@ref) and [`residuals`](@ref).

### Multivariate
