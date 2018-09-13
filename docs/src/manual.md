```@meta
DocTestSetup = quote
    using Random
    Random.seed!(1)
end
DocTestFilters = r".*[0-9\.]"
```

We will be using the data from Bollerslev and Ghysels (1996)[^1], available as the constant [`BG96`](@ref). The data consist of daily German mark/British pound exchange rates (1974 observations) and are often used in evaluating
implementations of (G)ARCH models[^2]. We begin by convincing ourselves that the data exhibit ARCH effects; a quick and dirty way of doing this is to look at the sample autocorrelation function of the squared returns:

```jldoctest MANUAL
julia> using ARCH

julia> data = BG96;

julia> autocor(data.^2, 1:10, demean=true)
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
Next, an instance of [`ARCHModel`](@ref) can be created by calling the constructor as follows:

[^1]: Bollerslev, T. and Ghysels, E. (1996), Periodic Autoregressive Conditional Heteroscedasticity, Journal of Business and Economic Statistics (14), pp. 139-151. [DOI: 10.2307/1392425](https://doi.org/10.2307/1392425)
[^2]: Brooks, C., Burke, S. P., and Persand, G. (2001), Benchmarks and the accuracy of GARCH model estimation, International Journal of Forecasting (17), pp. 45-56.[DOI: 10.1016/S0169-2070(00)00070-4](https://doi.org/10.1016/S0169-2070(00)00070-4)

```jldoctest MANUAL
julia> spec = GARCH{1, 1}([1., .9, .05]);

julia> am = ARCHModel(spec, data)

GARCH{1,1} model with Gaussian errors, T=1000.


               ω  β₁   α₁
Parameters:  1.0 0.9 0.05
```

Notice that no intercept is included by default, and a standard Normal distribution is assumed. The constructor accepts the keyword arguments `dist` and `meanspec` for specifying alternative distributions and mean specifications:

```jldoctest MANUAL
julia> am = ARCHModel(spec, data; dist=StdTDist(3.), meanspec=Intercept(1.))

GARCH{1,1} model with Student's t errors, T=1000.


               ω  β₁   α₁   ν   μ
Parameters:  1.0 0.9 0.05 3.0 1.0

```

The model can then be fitted as follows:

```jldoctest MANUAL
julia> fit!(am)

GARCH{1,1} model with Student's t errors, T=1974.


Mean equation parameters:

       Estimate  Std.Error  z value Pr(>|z|)
μ    0.00227251 0.00686802 0.330882   0.7407

Volatility parameters:

       Estimate  Std.Error z value Pr(>|z|)
ω    0.00232225 0.00163909 1.41679   0.1565
β₁     0.884488   0.036963  23.929   <1e-99
α₁     0.124866  0.0405471 3.07952   0.0021

Distribution parameters:

     Estimate Std.Error z value Pr(>|z|)
ν     4.11211  0.400384 10.2704   <1e-24
```

```@meta
DocTestSetup = nothing
DocTestFilters = nothing
```

It should rarely be necessary to call the constructor directly; typically, instances of [`ARCHModel`](@ref) are created by calling [`simulate`](@ref) or [`fit`](@ref).
