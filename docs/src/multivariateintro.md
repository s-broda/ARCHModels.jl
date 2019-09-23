# Introduction

ARCH models naturally generalize to the multivariate setting. Consider a time series of daily asset returns $\{r_t\}_{t\in 1, \ldots, T}$, where now  $r_t\in\mathbb{R}^d$. Similarly to the univariate case, the general model structure is

\[
r_t=\mu_t+\Sigma_t^{1/2}z_t,\quad \mu_t\equiv\mathbb{E}[r_t\mid\mathcal{F}_{t-1}],\quad \Sigma_t\equiv\mathbb{E}[(r_t-\mu_t)(r_t-\mu_t)^\mathrm{T}]\mid\mathcal{F}_{t-1}].
\]

A multivariate ARCH model specifies the conditional covariance matrix $\Sigma_t$ in terms of past returns, conditional (co)variances, and potentially other variables. The main challenge in multivariate ARCH modelling is the _curse of dimensionality_: allowing each of the $(d)(d+1)/2$ elements of $\Sigma_t$ to depend on the past returns of all $d$ other assets requires $O(d^4)$ parameters without imposing additional structure. Multivariate ARCH models differ in which structure they impose.

The following multivariate ARCH models are currently available in this package:

  * The CCC model of [Bollerslev (1990)](https://doi.org/10.2307/2109358)
  * The DCC model of [Engle (2002)](https://doi.org/10.1198/073500102288618487)

These may be combined with different mean specifications as in the univariate case, and (in principle) with different specifications for the joint distribution of the $z_t$, although at present, only the multivariate standard normal is supported. Multivariate ARCH models are represented as instances of [`MultivariateARCHModel`](@ref), which like [`UnivariateARCHModel`](@ref) subtypes [`ARCHModel`](@ref).

# Type hierarchy
## [Covariance specifications](@id covspec)
Volatility specifications describe the evolution of $\Sigma_t$. They are modelled as subtypes of [`MultivariateVolatilitySpec`](@ref).

### CCC
The CCC (and DCC, see below) models are examples of _conditional correlation_ models. They decompose
$\Sigma_t$ as
\[
\Sigma_t=D_t R_t D_t,
\]
where $R_t$ is the conditional correlation matrix and $D_t$ is a diagonal matrix containing the volatilities of the individual assets, which are modelled as univariate ARCH processes. In the constant conditional correlation (CCC) model, $R_t=R$ is assumed constant. The model is typically, including here, estimated in a two-step procedure: first, univariate ARCH models are fitted to the $d$ asset returns, and then $R$ is estimated as the sample correlation matrix of the standardized residuals.

### DCC
The DCC model extends the CCC model by making the $R_t$ dynamic (hence the name, dynamic conditional correlation model). In particular,

\[
R_{ij, t} = \frac{Q_{ij,t}}{\sqrt{Q_{ii,t}Q_{jj,t}}}, \
\]
where
\[Q_{t} =(1-\theta _{1}-\theta _{2})\bar{Q}+\theta _{1}\epsilon
_{t-1}\epsilon _{t-1}^{\prime }+\theta _{2}Q_{t-1},
\]
$\epsilon _{t}=D_{t}^{-1}a_{t}$, $Q_{t}=\mathrm{cov}%
(\epsilon _{t}|F_{t-1})$, and $\bar{Q}=\mathrm{cov}(\epsilon _{t})$.
