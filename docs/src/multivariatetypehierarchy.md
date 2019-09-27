```@meta
DocTestSetup = quote
    using ARCHModels    
end
```
# Type hierarchy: Multivariate
Analogously to the univariate case, an instance of [`MultivariateARCHModel`](@ref) contains a matrix of data (with observations in rows and assets in columns), and encapsulates information about the [covariance specification](@ref covspec) (e.g., [CCC](@ref) or [DCC](@ref)), the [mean specification](@ref mvmeanspec), and the [error distribution](@ref mvdistspec).

[`MultivariateARCHModel`](@ref)s support many of the same methods as [`UnivariateARCHModel`](@ref)s, with a few noteworthy differences: the prediction targets for [`predict`](@ref) are `:covariances` and `:correlations` for predicting ``\Sigma_t`` and ``R_t``, respectively, and the new functions [`covariances`](@ref) and [`correlations`](@ref) respectively return the in-sample estimates of ``\Sigma_t`` and ``R_t``.

## [Covariance specifications](@id covspec)
The dynamics of ``\Sigma_t``  are modelled as subtypes of [`MultivariateVolatilitySpec`](@ref).
### Conditional correlation models
The main challenge in multivariate ARCH modelling is the _curse of dimensionality_: allowing each of the ``(d)(d+1)/2`` elements of ``\Sigma_t`` to depend on the past returns of all ``d`` other assets requires ``O(d^4)`` parameters without imposing additional structure. Conditional correlation models approach this issue by decomposing
``\Sigma_t`` as
```math
\Sigma_t=D_t R_t D_t,
```
where ``R_t`` is the conditional correlation matrix and ``D_t`` is a diagonal matrix containing the volatilities of the individual assets, which are modelled as univariate ARCH processes.

#### DCC
The dynamic conditional correlation (DCC) model of [Engle (2002)](https://doi.org/10.1198/073500102288618487) imposes a GARCH-type structure on the ``R_t``. In particular, for a DCC(p, q) model (with covariance targeting),

```math
R_{ij, t} = \frac{Q_{ij,t}}{\sqrt{Q_{ii,t}Q_{jj,t}}},
```
where
```math
Q_{t} \equiv\bar{Q}(1-\bar\alpha-\bar\beta)+\sum_{i=1}^{p} \beta_iQ_{t-i}+\sum_{i=1}^{q}\alpha_i\epsilon_{t-i}\epsilon_{t-i}^\mathrm{\scriptsize T},
```
``\bar{\alpha}\equiv\sum_{i=1}^q\alpha_i``, ``\bar{\beta}\equiv\sum_{i=1}^q\beta_i``, ``\epsilon_{t}\equiv D_t^{-1}a_t$, $Q_{t}=\mathrm{cov}
(\epsilon_t|F_{t-1})``, and ``\bar{Q}=\mathrm{cov}(\epsilon_{t})``.

It is available as `DCC{p, q}`. The constructor takes as inputs ``\bar{Q}``, a vector of coefficients, and a vector of `UnivariateARCHModel`s:

```jldoctest
julia> DCC{1, 1}([1. .5; .5 1.], [.9, .05], [GARCH{1, 1}([1., .9, .05]) for _ in 1:2])
DCC{1, 1, TGARCH{0,1,1}} specification.

──────────────────────
              β₁    α₁
──────────────────────
Parameters:  0.9  0.05
──────────────────────
```

The DCC model is typically estimated in two steps, by first fitting univariate ARCH models to the individual assets and saving the standardized residuals ``\{\epsilon_t\}``, and then estimating the DCC parameters from those. [Engle (2002)](https://doi.org/10.1198/073500102288618487) provides the details and expressions for the standard errors. By default, this package employs an alternative estimator due to [Engle, Ledoit, and Wolf (2019)](https://doi.org/10.1080/07350015.2017.1345683) which is better suited to large-dimensional problems. It achieves this by i) estimating ``\bar{Q}`` with a nonlinear shrinkage estimator instead of the sample covariance of $\epsilon_t$, and ii) estimating the DCC parameters by maximizing the sum of the pairwise log-likelihoods, rather than the joint log-likelihood over all assets, thereby avoiding the inversion of large matrices during the optimization. The estimation method is controlled by passing the `method` keyword to the constructor. Possible values are `:largescale` (the default), and `:twostep`.

#### CCC
The CCC (constant conditional correlation) model of [Bollerslev (1990)](https://doi.org/10.2307/2109358) models ``R_t=R`` as constant. It is the special case of the DCC model in which ``p=q=0``:

```jldoctest
julia> CCC == DCC{0, 0}
true
```
As such, the constructor has the exact same signature, except that the DCC parameters must be passed as a zero-length vector:

```jldoctest
julia> CCC([1. .5; .5 1.], Float64[], [GARCH{1, 1}([1., .9, .05]) for _ in 1:2])
DCC{0, 0, TGARCH{0,1,1}} specification.

No estimable parameters.
```

As for the DCC model, the constructor accepts a `method` keyword argument with possible values `:largescale` (default) or `:twostep` that determines whether ``R`` will be estimated by nonlinear shrinkage or the sample correlation of the ``\epsilon_t``.

## [Mean Specifications](@id mvmeanspec)
 The conditional mean of a [`MultivariateARCHModel`](@ref) is specified by a vector of [`MeanSpec`](@ref)s as described under [Mean specifications](@ref meanspec).

## [Multivariate Standardized Distributions](@id mvdistspec)
Multivariate standardized distributions subtype [`MultivariateStandardizedDistribution`](@ref). Currently, only [`MultivariateStdNormal`](@ref) is available. Note that under mild assumptions, the Gaussian (quasi-)MLE consistently estimates the (multivariate) ARCH parameters even if Gaussianity is violated.

```@meta
DocTestSetup = nothing
DocTestFilters = nothing
```
