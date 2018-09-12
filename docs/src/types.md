The basic type provided by this package is [`ARCHModel`](@ref). It implements the interface of `StatisticalModel` from [`StatsBase`](http://juliastats.github.io/StatsBase.jl/stable/statmodels.html). An instance of this type contains a vector of data (such as equity returns), and encapsulates information about the [volatility specification](@ref volaspec) (e.g., [GARCH](@ref) or [EGARCH](@ref)), the [mean specification](@ref meanspec) (e.g., whether an intercept is included), and the [error distribution](@ref Distributions).

# [Volatility specifications](@id volaspec)
Volatility specifications are modelled as subtypes of [`VolatilitySpec`](@ref). There is one type for each class of (G)ARCH model, parameterized by numbers of lags.
## GARCH
The GARCH(p, q) model specifies the volatility as
```math
\sigma_t^2=\omega+\sum_{i=1}^p\beta_i \sigma_{t-i}^2+\sum_{i=1}^q\alpha_i r_{t-i}^2, \quad \omega, \alpha, \beta>0,\quad \alpha+\beta<1.
```
The corresponding type in Julia is [`GARCH`](@ref). For example, a GARCH(1, 1) model with ``ω=1``, ``β=.9``, and ``α=.05`` is obtained with
```jldoctest TYPES
julia> using ARCH

julia> GARCH{1, 1}([1., .9, .05])
GARCH{1,1} specification.

               ω  β₁   α₁
Parameters:  1.0 0.9 0.05
```

## EGARCH
# [Mean specifications](@id meanspec)
# Distributions
