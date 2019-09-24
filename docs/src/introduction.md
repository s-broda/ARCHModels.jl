# Introduction
Consider a sample of daily asset returns ``\{r_t\}_{t\in\{1,\ldots,T\}}``. All models covered in this package share the same basic structure, in that they decompose the return into a conditional mean and a mean-zero innovation. In the univariate case,
```math
r_t=\mu_t+a_t,\quad \mu_t\equiv\mathbb{E}[r_t\mid\mathcal{F}_{t-1}],\quad \sigma_t^2\equiv\mathbb{E}[a_t^2\mid\mathcal{F}_{t-1}],
```
``z_t\equiv a_t/\sigma_t`` is identically and independently distributed according to some law with mean zero and unit variance, and ``\\{\mathcal{F}_t\\}`` is the natural filtration of ``\\{r_t\\}`` (i.e., it encodes information about past returns). In the multivariate case, ``r_t\in\mathbb{R}^d``, and the general model structure is

```math
r_t=\mu_t+a_t,\quad \mu_t\equiv\mathbb{E}[r_t\mid\mathcal{F}_{t-1}],\quad \Sigma_t\equiv\mathbb{E}[a_ta_t^\mathrm{\scriptsize T}]\mid\mathcal{F}_{t-1}].
```

ARCH models specify the conditional volatility ``\sigma_t`` (or in the multivariate case, the conditional covariance matrix ``\Sigma_t``) in terms of past returns, conditional (co)variances, and potentially other variables.


This package represents an ARCH model as an instance of either [`UnivariateARCHModel`](@ref) or [`MultivariateARCHModel`](@ref). These are subtypes [`ARCHModel`](@ref) and implement the interface of `StatisticalModel` from [`StatsBase`](http://juliastats.github.io/StatsBase.jl/stable/statmodels.html).
