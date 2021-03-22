# The ARCHModels Package
ARCH (Autoregressive Conditional Heteroskedasticity) models are a class of models designed to capture a feature of financial returns data known as *volatility clustering*, *i.e.*, the fact that large (in absolute value) returns tend to cluster together, such as during periods of financial turmoil, which then alternate with relatively calmer periods.

The basic ARCH model was introduced by Engle (1982, Econometrica, pp. 987–1008), who in 2003 was awarded a Nobel Memorial Prize in Economic Sciences for its development. Today, the most popular variant is the generalized ARCH, or GARCH, model and its various extensions, due to Bollerslev (1986, Journal of Econometrics, pp. 307 - 327). The basic GARCH(1,1) model for a sample of daily asset returns ``\{r_t\}_{t\in\{1,\ldots,T\}}`` is

```math
r_t=\sigma_tz_t,\quad z_t\sim\mathrm{N}(0,1),\quad
\sigma_t^2=\omega+\alpha r_{t-1}^2+\beta \sigma_{t-1}^2,\quad \omega, \alpha, \beta>0,\quad \alpha+\beta<1.
```

This can be extended by including additional lags of past squared returns and volatilities: the GARCH(p, q) model  has ``q`` of the former and ``p`` of the latter. Another generalization is to allow  ``z_t`` to follow other, non-Gaussian distributions.

This package implements simulation, estimation, and model selection for the following univariate models:

  * ARCH(q)
  * GARCH(p, q)
  * TGARCH(o, p, q)
  * EGARCH(o, p q)

The conditional mean can be specified as either zero, an intercept, a linear regression model, or an ARMA(p, q) model.
As for error distributions, the user may choose among the following:

  * Standard Normal
  * Standardized Student's ``t``
  * Standardized Hansen Skewed ``t``
  * Standardized Generalized Error Distribution

For instance, a GARCH(1,1) model with a conditional mean from an AR(1) model with normally distributed errors can be esimated by
`fit(GARCH{1,1}, data; meanspec=AR{1}, dist=StdNormal)`.

In addition, the following multivariate models are supported:

  * CCC
  * DCC(p, q)

## Installation

`ARCHModels` is a registered Julia package. To install it in Julia 1.0 or later, do

```
add ARCHModels
```

in the Pkg REPL mode (which is entered by pressing `]` at the prompt).

## Acknowledgements

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 750559.

![EU LOGO](assets/EULOGO.jpg)
