```@meta
DocTestSetup = quote
    using Random
    Random.seed!(1)
end
DocTestFilters = r".*[0-9\.]"
```

We will be using the data from [Bollerslev and Ghysels (1986)](https://doi.org/10.2307/1392425), available as the constant [`BG96`](@ref). The data consist of daily German mark/British pound exchange rates (1974 observations) and are often used in evaluating
implementations of (G)ARCH models (see, e.g., [Brooks et.al. (2001)](https://doi.org/10.1016/S0169-2070(00)00070-4). We begin by convincing ourselves that the data exhibit ARCH effects; a quick and dirty way of doing this is to look at the sample autocorrelation function of the squared returns:

```jldoctest MANUAL
julia> using ARCH

julia> data = BG96;

julia> autocor(data.^2, 1:10, demean=true) # re-exported from StatsBase
10-element Array{Float64,1}:
 0.22294073831639766
 0.17663183540117078
 0.14086005904595456
 0.1263198344036979
 0.18922204038617135
 0.09068404029331875
 0.08465365332525085
 0.09671690899919724
 0.09217329577285414
 0.11984168975215709
```

Using a critical value of ``1.96/\\sqrt{1974}=0.044``, we see that there is indeed significant autocorrelation in the squared series.

# Estimation
Having established the presence of volatility clustering, we can begin by fitting the workhorse model of volatility modeling, a GARCH(1, 1) with standard normal errors;  for other model classes such as [`EGARCH`](@ref), see the [section on volatility specifications](@ref volaspec).

```
julia> fit(GARCH{1, 1}, data)

GARCH{1,1} model with Gaussian errors, T=1974.


Mean equation parameters:

        Estimate  Std.Error   z value Pr(>|z|)
μ    -0.00616637 0.00920163 -0.670139   0.5028

Volatility parameters:

      Estimate  Std.Error z value Pr(>|z|)
ω    0.0107606 0.00649493 1.65677   0.0976
β₁    0.805875  0.0725003 11.1155   <1e-27
α₁    0.153411  0.0536586 2.85903   0.0042
```

This returns an instance of [`ARCHModel`](@ref), as described in the section [Working with ARCHModels](@ref). The parameters ``\alpha_1`` and ``\beta_1`` in the volatility equation are highly significant, again confirming the presence of volatility clustering. Note also that the fitted values are the same as those found by [Bollerslev and Ghysels (1986)](https://doi.org/10.2307/1392425) and [Brooks et.al. (2001)](https://doi.org/10.1016/S0169-2070(00)00070-4) for the same dataset.

The [`fit`](@ref) method supports a number of keyword arguments; the full signature is
```julia
fit(::Type{<:VolatilitySpec}, data::Vector; dist=StdNormal, meanspec=Intercept, algorithm=BFGS(), autodiff=:forward, kwargs...)
```

Their meaning is as follows:
- `dist`: the error distribution. A subtype (*not instance*) of [`StandardizedDistribution`](@ref); see Section [Distributions](@ref).
- `meanspec=Intercept`: the mean specification. A subtype of [`MeanSpec`](@ref); see the [section on mean specification](@ref meanspec).
The remaining keyword arguments are passed on to the optimizer.

As an example, an EGARCH(1, 1, 1) model without intercept and with  Student's ``t`` errors is fitted as follows:

```jldoctest MANUAL
julia> fit(EGARCH{1, 1, 1}, data; meanspec=NoIntercept, dist=StdTDist)

EGARCH{1,1,1} model with Student's t errors, T=1974.


Volatility parameters:

       Estimate Std.Error   z value Pr(>|z|)
ω    -0.0162014 0.0186806 -0.867286   0.3858
γ₁   -0.0378454  0.018024  -2.09972   0.0358
β₁     0.977687  0.012558   77.8538   <1e-99
α₁     0.255804 0.0625497   4.08961    <1e-4

Distribution parameters:

     Estimate Std.Error z value Pr(>|z|)
ν     4.12423   0.40059 10.2954   <1e-24
```

An alternative approach to fitting a [`VolatilitySpec`](@ref) to `data` is to first construct
an [`ARCHModel`](@ref) containing the data, and then using [`fit!`](@ref) to modify it in place:

```jldoctest MANUAL
julia> am = ARCHModel(GARCH{1, 1}([1., 0., 0.]), data)

GARCH{1,1} model with Gaussian errors, T=1974.


                             ω  β₁  α₁
Volatility parameters:     1.0 0.0 0.0



julia> fit!(am)

GARCH{1,1} model with Gaussian errors, T=1974.


Volatility parameters:

      Estimate  Std.Error z value Pr(>|z|)
ω    0.0108661 0.00657449 1.65277   0.0984
β₁    0.804431  0.0730395 11.0136   <1e-27
α₁    0.154597  0.0539319 2.86651   0.0042
```

Calling `fit(am)` will return a new instance of ARCHModel instead:

```jldoctest MANUAL
julia> am2 = fit(am);

julia> am2 === am
false

julia> am2.spec.coefs == am.spec.coefs
true
```

# Model selection
The function [`selectmodel`](@ref) can be used for automatic model selection, based on an information crititerion. Given
a class of model (i.e., a subtype of [`VolatilitySpec`](@ref)), it will return a fitted [`ARCHModel`](@ref), with the lag length
parameters (i.e., ``p`` and ``q`` in the case of [`GARCH`](@ref)) chosen to minimize the desired criterion. The [BIC](https://en.wikipedia.org/wiki/Bayesian_information_criterion) is used by default.

Eg., the following selects the optimal (minimum AIC) EGARCH(o, p, q) model, where o, p, q < 2,  assuming ``t`` distributed errors.

```jldoctest MANUAL
julia> selectmodel(EGARCH, data; criterion=aic, maxlags=2, dist=StdTDist)

EGARCH{1,1,2} model with Student's t errors, T=1974.


Mean equation parameters:

       Estimate  Std.Error  z value Pr(>|z|)
μ    0.00196126 0.00695292 0.282077   0.7779

Volatility parameters:

       Estimate Std.Error   z value Pr(>|z|)
ω    -0.0031274 0.0112456 -0.278101   0.7809
γ₁   -0.0307681 0.0160754  -1.91398   0.0556
β₁     0.989056 0.0073654   134.284   <1e-99
α₁     0.421644 0.0678139   6.21767    <1e-9
α₂    -0.229068 0.0755326   -3.0327   0.0024

Distribution parameters:

     Estimate Std.Error z value Pr(>|z|)
ν     4.18795  0.418697 10.0023   <1e-22
```

Passing the keyword argument `show_trace=false` will show the criterion for each model after it is estimated.

# Risk measures
One of the primary uses of ARCH models is for estimating and forecasting risk measures, such as [Value at Risk](https://en.wikipedia.org/wiki/Value_at_risk) and [Expected Shortfall](https://en.wikipedia.org/wiki/Expected_shortfall).
This section details the relevant functionality provided in this package.

Basic in-sample estimates for the Value at Risk implied by an estimated [`ARCHModel`](@ref) can be obtained using [`VaRs`](@ref):

```jldoctest MANUAL
julia> am = fit(GARCH{1, 1}, BG96);

julia> VaRs(am)[end]
0.7945179524273573
```

# Forecasting
The [`predict(am::ARCHModel)`](@ref) method can be used to construct one-step ahead forecasts for a number of quantities. Its signature is
```
    predict(am::ARCHModel, what=:volatility; level=0.01)
```
The keyword argument `what` controls which object is predicted;
the choices are `:volatility` (the default), `:variance`, `:return`, and `:VaR`. The VaR level can be controlled with the keyword argument `level`.

# Model diagnostics and specification tests
Testing volatility models in general relies on the estimated conditional volatilities ``\hat{\sigma}_t`` and the standardized residuals
``\hat{z}_t\equiv (r_t-\hat{\mu}_t)/\hat{\sigma}_t``, accessible via [`volatilities(::ARCHModel)`](@ref) and [`residuals(::ARCHModel)`](@ref), respectively. The non-standardized
residuals ``\hat{u}_t\equiv r_t-\hat{\mu}_t`` can be obtained by passing `standardized=false` as a keyword argument to [`residuals`](@ref).

# Simulation
To simulate from an [`ARCHModel`](@ref), use [`simulate`](@ref). You can either specify the [`VolatilitySpec`](@ref) (and optionally the distribution and mean specification) and desired number of observations, or pass an existing [`ARCHModel`](@ref). Use [`simulate!`](@ref) to modify the data in place.

```jldoctest MANUAL
julia> am3 = simulate(GARCH{1, 1}([1., .9, .05]), 1000; warmup=500, meanspec=Intercept(5.), dist=StdTDist(3.))

GARCH{1,1} model with Student's t errors, T=1000.


                             μ
Mean equation parameters:  5.0

                             ω  β₁   α₁
Volatility parameters:     1.0 0.9 0.05

                             ν
Distribution parameters:   3.0

julia> am4 = simulate(am3);
```
```@meta
DocTestSetup = nothing
DocTestFilters = nothing
```
