# Introduction
Consider a sample of daily asset returns ``\{r_t\}_{t\in\{1,\ldots,T\}}``. All models covered in this package share the same basic structure, in that they decompose the return into a conditional mean and a mean-zero innovation:
```math
r_t=\mu_t+a_t,\quad \mu_t\equiv\mathbb{E}[r_t\mid\mathcal{F}_{t-1}],\quad \sigma_t^2\equiv\mathbb{E}[a_t^2\mid\mathcal{F}_{t-1}],
```
and ``z_t\equiv a_t/\sigma_t`` is identically and independently distributed according to some law with mean zero and unit variance and ``\\{\mathcal{F}_t\\}`` is the natural filtration of ``\\{r_t\\}`` (i.e., it encodes information about past returns).


ARCH models naturally generalize to the multivariate setting. Consider again a time series of daily asset returns ``\{r_t\}_{t\in 1, \ldots, T}``, where now  ``r_t\in\mathbb{R}^d``. Similarly to the univariate case, the general model structure is

```math
r_t=\mu_t+a_t,\quad \mu_t\equiv\mathbb{E}[r_t\mid\mathcal{F}_{t-1}],\quad \Sigma_t\equiv\mathbb{E}[a_ta_t^\mathrm{\scriptsize T}]\mid\mathcal{F}_{t-1}].
```

A multivariate ARCH model specifies the conditional covariance matrix ``\Sigma_t`` in terms of past returns, conditional (co)variances, and potentially other variables.


This package represents an ARCH model as an instance of either [`UnivariateARCHModel`](@ref) or [`MultivariateARCHModel`](@ref). These are subtypes [`ARCHModel`](@ref) and implement the interface of `StatisticalModel` from [`StatsBase`](http://juliastats.github.io/StatsBase.jl/stable/statmodels.html). An instance of [`UnivariateARCHModel`](@ref) contains a vector of data (such as equity returns), and encapsulates information about the [volatility specification](@ref volaspec) (e.g., [GARCH](@ref) or [EGARCH](@ref)), the [mean specification](@ref meanspec) (e.g., whether an intercept is included), and the [error distribution](@ref Distributions). 
