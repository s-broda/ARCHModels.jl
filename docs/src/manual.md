# Manual
## Introduction
The basic type provided by this package is [`ARCHModel`](@ref). It implements the interface of `StatisticalModel` from [`StatsBase`](http://juliastats.github.io/StatsBase.jl/stable/statmodels.html). An instance of this type contains a vector of data (such as equity returns), and encapsulates information about the [volatility specification](@ref volaspec) (e.g., [GARCH](@ref) or [EGARCH](@ref)), the [mean specification](@ref meanspec) (e.g., whether an intercept is included), and the [error distribution](@ref Distributions).

An instance of `ARCHModel` can be created directly via its [constructor](@ref ARCHModel(spec::VolatilitySpec, data::Vector{<:AbstractFloat}).
## [Volatility specifications](@id volaspec)
### GARCH
### EGARCH
## [Mean specifications](@id meanspec)
## Distributions
