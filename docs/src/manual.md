```@meta
DocTestSetup = quote
    using Random
    Random.seed!(1)
end
DocTestFilters = r".*[0-9\.]"
```
An instance of [ARCHModel](@ref) can be created by calling the constructor as follows:

```jldoctest CONSTRUCTOR
julia> spec = GARCH{1, 1}([1., .9, .05])
GARCH{1,1,Float64}([1.0, 0.9, 0.05])

julia> spec = GARCH{1, 1}([1., .9, .05]);

julia> data = randn(1_000);

julia> am = ARCHModel(spec, data)

GARCH{1,1} model with Gaussian errors, T=1000.


               ω  β₁   α₁
Parameters:  1.0 0.9 0.05
```

Notice that no intercept is included by default, and a standard Normal distribution is assumed. The constructor accepts the keyword arguments `dist` and `meanspec` for specifying alternative distributions and mean specifications:

```jldoctest CONSTRUCTOR
julia am = ARCHModel(spec, data; dist=StdTDist(3.), meanspec=Intercept(1.))

GARCH{1,1} model with Student's t errors, T=1000.


               ω  β₁   α₁   ν   μ
Parameters:  1.0 0.9 0.05 3.0 1.0

```

The model can then be fitted as follows:

```jldoctest CONSTRUCTOR
julia> fit!(am)

GARCH{1,1} model with Student's t errors, T=1000.


Mean equation parameters:

       Estimate Std.Error   z value Pr(>|z|)
μ    -0.0289844 0.0316589 -0.915519   0.3599

Volatility parameters:

       Estimate  Std.Error  z value Pr(>|z|)
ω     0.0420412  0.0390976  1.07529   0.2822
β₁     0.952037  0.0398453  23.8933   <1e-99
α₁   0.00591995 0.00835634 0.708438   0.4787

Distribution parameters:

     Estimate Std.Error  z value Pr(>|z|)
ν      99.936    271.25 0.368428   0.7126
```

Recall that the data in this example are i.d.d. normal, so the estimation results aren't very meaningful.

 Instances of [ARCHModel](@ref) are also returned from [`simulate`](@ref) and [`fit`](@ref).
 ```@meta
 DocTestSetup = nothing
 DocTestFilters = nothing
 ```
