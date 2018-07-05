# The ARCH Package

ARCH (Autoregressive Conditional Heteroskedasticty) models are a class of models designed to capture a feature of financial returns data known as *volatility clustering*, *i.e.*, the fact that large (in absolute value) returns tend to cluster together, such as during periods of financial turmoil, which then alternate with relatively calmer periods.

The basic ARCH model was introduced by Engle (1982, Econometrica, pp. 987–1008), who in 2003 was awarded a Nobel Memorial Prize in Economic Sciences for its development. Today, the most popular variant is the generalized ARCH, or GARCH, model and its various extensions, due to Bollerslev (1986, Journal of Econometrics, pp. 307 - 327). The basic GARCH(1,1) model for a sample of daily asset returns ``\{r_t\}_{t\in\{1,\ldots,T\}}`` is

```math r_t=\sigma_tz_t,\quad z_t\sim\mathrm{N}(0,1),\quad
\sigma_t^2=\omega+\alpha r_{t-1}^2+\beta \sigma_{t-1}^2,\quad \omega, \alpha, \beta>0,\quad \alpha+\beta<1.
```

This can be extended by including additional lags of past squared returns and volatilities: the GARCH(p, q) model  has ``q`` of the former and ``p`` of the latter.

## Contents
```@contents
Depth = 2
```

## Acknowledgements

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 750559).

<img src="assets/LOGO_ERC-FLAG_EU_.jpg" width="240">
