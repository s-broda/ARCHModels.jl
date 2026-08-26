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
ARMA{1, 1, Float64}([1.0, 0.9, -0.1])
```
Pure AR(p) and MA(q) models are obtained as follows:
```jldoctest TYPES
julia> AR{1}([1., .9])
AR{1, Float64}([1.0, 0.9])
julia> MA{1}([1., -.1])
MA{1, Float64}([1.0, -0.1])
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
