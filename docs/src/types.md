Consider a sample of daily asset returns ``\\{r_t\\}_{t\in\{1,\ldots,T\}}``. All models covered in this package share the same basic structure, in that they decompose the return into a conditional mean and a mean-zero innovation:
```math
r_t=\mu_t+\sigma_tz_t,\quad \mu_t\equiv\mathbb{E}[r_t\mid\mathcal{F}_{t-1}],\quad \sigma_t^2\equiv\mathbb{E}[(r_t-\mu_t)^2\mid\mathcal{F}_{t-1}],
```
where ``z_t`` is identically and independently distributed according to some law with mean zero and unit variance and ``\\{\mathcal{F}_t\\}`` is the natural filtration of ``\\{r_t\\}`` (i.e., it encodes information about past returns).

This package represents a (G)ARCH model as an instance of [`ARCHModel`](@ref), which implements the interface of `StatisticalModel` from [`StatsBase`](http://juliastats.github.io/StatsBase.jl/stable/statmodels.html). An instance of this type contains a vector of data (such as equity returns), and encapsulates information about the [volatility specification](@ref volaspec) (e.g., [GARCH](@ref) or [EGARCH](@ref)), the [mean specification](@ref meanspec) (e.g., whether an intercept is included), and the [error distribution](@ref Distributions).

# [Volatility specifications](@id volaspec)
Volatility specifications describe the evolution of ``\sigma_t``. They are modelled as subtypes of [`VolatilitySpec`](@ref). There is one type for each class of (G)ARCH model, parameterized by numbers of lags.
## GARCH
The GARCH(p, q) model, due to [Bollerslev (1986)](https://doi.org/10.1016/0304-4076(86)90063-1) specifies the volatility as
```math
\sigma_t^2=\omega+\sum_{i=1}^p\beta_i \sigma_{t-i}^2+\sum_{i=1}^q\alpha_i r_{t-i}^2, \quad \omega, \alpha_i, \beta_i>0,\quad \sum_{i=1}^{\max p,q} \alpha_i+\beta_i<1.
```
The corresponding type is [`GARCH{p, q}`](@ref). For example, a GARCH(1, 1) model with ``ω=1``, ``β=.9``, and ``α=.05`` is obtained with
```jldoctest TYPES
julia> using ARCH

julia> GARCH{1, 1}([1., .9, .05])
GARCH{1,1} specification.

               ω  β₁   α₁
Parameters:  1.0 0.9 0.05
```

As for all subtypes of [`VolatilitySpec`](@ref), the order of the parameters in the coefficient vector is such that all parameters pertaining to the first type parameter `p` (corresponding to the first sum in the equation) appear before those pertaining to the second, `q`.

As a special case, the ARCH(q) volatility specification, due to [Engle (1982)](https://doi.org/10.2307/1912773 ), is
```math
\sigma_t^2=\omega+\sum_{i=1}^q\alpha_i r_{t-i}^2,
```
corresponding to a GARCH{0, q} model. It is available as [`_ARCH{q}`](@ref):
```jldoctest TYPES
julia> _ARCH{2}([1., .5, .4])
GARCH{0,2} specification.

               ω  α₁  α₂
Parameters:  1.0 0.5 0.4
```
## EGARCH
The EGARCH{o, p, q} volatility specification, due to [Nelson (1991)](https://doi.org/10.2307/2938260), is
```math
\log(\sigma_t^2)=\omega+\sum_{i=1}^o\gamma_i z_{t-i}+\sum_{i=1}^p\beta_i \log(\sigma_{t-i}^2)+\sum_{i=1}^q\alpha_i (|z_{t-i}|-\sqrt{2/\pi}), \quad z_t=r_t/\sigma_t,\quad \sum_{i=1}^{p}\beta_i<1.
```

The corresponding type is [`EGARCH{o, p, q}`](@ref):
```jldoctest TYPES
julia> EGARCH{1, 1, 1}([-0.1, .1, .9, .04])
EGARCH{1,1,1} specification.

                ω  γ₁  β₁   α₁
Parameters:  -0.1 0.1 0.9 0.04
```
# [Mean specifications](@id meanspec)
Mean specifications serve to specify ``\mu_t``. They are modelled as subtypes of [`MeanSpec`](@ref). They contain their parameters as (possibly empty) vectors, but convenience constructors are provided where appropriate. Currently, two specifications are available:
* ``\mu_t=0``, available as [`NoIntercept`](@ref):
```jldoctest TYPES
julia> NoIntercept() # convenience constructor, eltype defaults to Float64
NoIntercept{Float64}(Float64[])
```
* ``\mu_t=\mu``, available as [`Intercept`](@ref):
```jldoctest TYPES
julia> Intercept(3) # convenience constructor
Intercept{Float64}([3.0])
```
# Distributions
## Built-in distributions
Different distributions of ``z_t`` are available as subtypes of [`StandardizedDistribution`](@ref). `StandardizedDistribution` in turn subtypes `Distribution{Univariate, Continuous}` from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl), though not the entire interface must necessarily be implemented. `StandardizedDistribution`s again hold their parameters as vectors, but convenience constructors are provided. The following are currently available:
* [`StdNormal`](@ref), the standard normal distribution:
```jldoctest TYPES
julia> StdNormal() # convenience constructor
StdNormal{Float64}(coefs=Float64[])
```
* [`StdTDist`](@ref), the standardized Student's `t` distribution:
```jldoctest TYPES
julia> StdTDist(3) # convenience constructor: 3 degrees of freedom
StdTDist{Float64}(coefs=[3.0])
```

## User-defined standardized distributions
Apart from the natively supported standardized distributions, it is possible to wrap a continuous univariate distribution from the [Distributions package](https://github.com/JuliaStats/Distributions.jl) in the [`Standardized`](@ref) wrapper type. Below, we reimplement the standardized normal distribution:

```jldoctest TYPES
julia> using Distributions

julia> const MyStdNormal = Standardized{Normal};
```

`MyStdNormal` can be used whereever a built-in distribution could, albeit with a speed penalty. Note also that if the underlying distribution (such as `Normal` in the example above) contains location and/or scale parameters, then these are no longer identifiable, which implies that the estimated covariance matrix of the estimators will be singular.

A final remark concerns the domain of the parameters: the estimation process relies on a starting value for the parameters of the distribution, say ``\theta\equiv(\theta_1, \ldots, \theta_p)'``. For a distribution wrapped in [`Standardized`](@ref), the starting value for ``\theta_i`` is taken to be a small positive value ϵ. This will fail if ϵ is not in the domain of ``\theta_i``; as an example, the standardized Student's ``t`` distribution is only defined for degrees of freedom larger than 2, because a finite variance is required for standardization. In that case, it is necessary to define a method of the (non-exported) function `startingvals` that returns a feasible vector of starting values, as follows:

```jldoctest TYPES
julia> const MyStdTDist = Standardized{TDist};

julia> ARCH.startingvals(::Type{<:MyStdTDist}, data::Vector{T}) where T = T[3.]
```
# Working with ARCHModels
The constructor for [`ARCHModel`](@ref) takes two mandatory arguments: an instance of a subtype of [`VolatilitySpec`](@ref), and a vector of returns. The mean specification and error distribution can be changed via the keyword arguments `meanspec` and `dist`, which respectively default to `NoIntercept` and `StdNormal`.

For example, to construct a GARCH(1, 1) model with an intercept and ``t``-distributed errors, one would do
```jldoctest TYPES
julia> spec = GARCH{1, 1}([1., .9, .05]);

julia> data = BG96;

julia> am = ARCHModel(spec, data; dist=StdTDist(3.), meanspec=Intercept(1.))

GARCH{1,1} model with Student's t errors, T=1974.


                             μ
Mean equation parameters:  1.0

                             ω  β₁   α₁
Volatility parameters:     1.0 0.9 0.05

                             ν
Distribution parameters:   3.0
```

The model can then be fitted as follows:

```jldoctest TYPES
julia> fit!(am)

GARCH{1,1} model with Student's t errors, T=1974.


Mean equation parameters:

       Estimate  Std.Error  z value Pr(>|z|)
μ    0.00227251 0.00686802 0.330882   0.7407

Volatility parameters:

       Estimate  Std.Error z value Pr(>|z|)
ω    0.00232225 0.00163909 1.41679   0.1565
β₁     0.884488   0.036963  23.929   <1e-99
α₁     0.124866  0.0405471 3.07952   0.0021

Distribution parameters:

     Estimate Std.Error z value Pr(>|z|)
ν     4.11211  0.400384 10.2704   <1e-24
```

It should, however, rarely be necessary to construct an `ARCHModel` manually via its constructor; typically, instances of it are created by calling [`fit`](@ref), [`selectmodel`](@ref), or [`simulate`](@ref).

As discussed earlier, [`ARCHModel`](@ref) implements the interface of StatisticalModel from [`StatsBase`](http://juliastats.github.io/StatsBase.jl/stable/statmodels.html), so you
can call `coef`, `coefnames`, `confint`, `dof`, `informationmatrix`, `isfitted`, `loglikelihood`, `nobs`,  `score`, `stderror`, `vcov`, etc. on its instances:

```jldoctest TYPES
julia> nobs(am)
1974
```

Other useful methods include [`volatilities`](@ref) and [`residuals`](@ref).
