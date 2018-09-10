# Quick Start Guide
```@meta
DocTestSetup = quote
    using Random
    Random.seed!(1)
end
DocTestFilters = r".*[0-9\.]"
```
## Simulate a GARCH{1, 1} model

```jldoctest GARCH
julia> using ARCH

julia> am = simulate(GARCH{1, 1}([1., .9, .05]), 10^4)

GARCH{1,1} model with Gaussian errors, T=10000.


               ω  β₁   α₁
Parameters:  1.0 0.9 0.05
```

`am` is of type `ARCHModel`:

```jldoctest GARCH
julia> typeof(am)
ARCHModel{Float64,GARCH{1,1,Float64},StdNormal{Float64},NoIntercept{Float64}}
```

Note that by default, the simulated model does not contain an intercept. `ARCHModel` implements
the interface of StatisticalModel from [`StatsBase`](http://juliastats.github.io/StatsBase.jl/stable/statmodels.html):



```jldoctest GARCH
julia> nobs(am)
10000
```

## Fit a GARCH{1, 1} model to the simulated data

```jldoctest GARCH
julia> fit(GARCH{1, 1}, am.data)

GARCH{1,1} model with Gaussian errors, T=10000.


Mean equation parameters:

      Estimate Std.Error  z value Pr(>|z|)
μ    0.0277078 0.0435526 0.636191   0.5247

Volatility parameters:

      Estimate  Std.Error z value Pr(>|z|)
ω     0.910479   0.146171 6.22886    <1e-9
β₁    0.905417  0.0103798  87.229   <1e-99
α₁   0.0503472 0.00523329 9.62057   <1e-21
```

Note that an intercept is included by default. If no intercept is desired, then the keyword argument `meanspec=NoIntercept` should be passed.
Alternatively, use `fit!` to modify `am` (which does not include an intercept) in-place:

```jldoctest GARCH
julia> fit!(am)

GARCH{1,1} model with Gaussian errors, T=10000.


Volatility parameters:

      Estimate  Std.Error z value Pr(>|z|)
ω     0.908648   0.145787 6.23269    <1e-9
β₁    0.905532  0.0103563 87.4379   <1e-99
α₁   0.0503246 0.00522825 9.62552   <1e-21
```

## Simulate and fit an EGARCH model with an intercept, assuming Student's t errors

```jldoctest GARCH
julia> am = simulate(EGARCH{1, 1, 1}([.1, 0., .9, .1]), 10^4; meanspec=Intercept(1.), dist=StdTDist(3.))

EGARCH{1,1,1} model with Student's t errors, T=10000.


               ω  γ₁  β₁  α₁   ν   μ
Parameters:  0.1 0.0 0.9 0.1 3.0 1.0



julia> fit!(am)

EGARCH{1,1,1} model with Student's t errors, T=10000.


Mean equation parameters:

     Estimate Std.Error z value Pr(>|z|)
μ     1.00043 0.0105728 94.6221   <1e-99

Volatility parameters:

       Estimate Std.Error  z value Pr(>|z|)
ω     0.0987938 0.0250802  3.93912    <1e-4
γ₁   0.00367806 0.0107783 0.341247   0.7329
β₁      0.90718 0.0248733   36.472   <1e-99
α₁     0.105632 0.0181307  5.82614    <1e-8

Distribution parameters:

     Estimate Std.Error z value Pr(>|z|)
ν     2.93074 0.0962429 30.4515   <1e-99
```

## Model selection
The function `selectmodel` can be used to determine the optimal lag length within a class of models:

```jldoctest GARCH
julia> selectmodel(EGARCH, am.data; meanspec=Intercept, dist=StdTDist)

EGARCH{1,1,1} model with Student's t errors, T=10000.


Mean equation parameters:

     Estimate Std.Error z value Pr(>|z|)
μ     1.00043 0.0105728 94.6221   <1e-99

Volatility parameters:

       Estimate Std.Error  z value Pr(>|z|)
ω     0.0987938 0.0250802  3.93912    <1e-4
γ₁   0.00367806 0.0107783 0.341247   0.7329
β₁      0.90718 0.0248733   36.472   <1e-99
α₁     0.105632 0.0181307  5.82614    <1e-8

Distribution parameters:

     Estimate Std.Error z value Pr(>|z|)
ν     2.93074 0.0962429 30.4515   <1e-99
```

By default, `selectmodel` minimizes the [BIC](https://en.wikipedia.org/wiki/Bayesian_information_criterion). In this case, the optimal lag length corresponds to the "truth"; viz., the one used to simulate the model.

```@meta
DocTestSetup = nothing
DocTestFilters = nothing
```
