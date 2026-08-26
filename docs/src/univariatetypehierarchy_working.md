## Working with UnivariateARCHModels
The constructor for [`UnivariateARCHModel`](@ref) takes two mandatory arguments: an instance of a subtype of [`UnivariateVolatilitySpec`](@ref), and a vector of returns. The mean specification and error distribution can be changed via the keyword arguments `meanspec` and `dist`, which respectively default to `NoIntercept` and `StdNormal`.

For example, to construct a GARCH(1, 1) model with an intercept and ``t``-distributed errors, one would do
```jldoctest TYPES
julia> spec = GARCH{1, 1}([1., .9, .05]);

julia> data = BG96;

julia> am = UnivariateARCHModel(spec, data; dist=StdT(3.), meanspec=Intercept(1.))

GARCH{1, 1} model with Student's t errors, T=1974.


──────────────────────────────
                             μ
──────────────────────────────
Mean equation parameters:  1.0
───────────────────────────────────────────────────────────────────────
                             ω   β₁    α₁
─────────────────────────────────────────
Volatility parameters:     1.0  0.9  0.05
───────────────────────────────────────────────────────────────────────
                             ν
──────────────────────────────
Distribution parameters:   3.0
──────────────────────────────
```

The model can then be fitted as follows:

```jldoctest TYPES
julia> fit!(am)

GARCH{1, 1} model with Student's t errors, T=1974.

Mean equation parameters:
─────────────────────────────────────────────
     Estimate   Std.Error   z value  Pr(>|z|)
─────────────────────────────────────────────
μ  0.00227251  0.00686802  0.330882    0.7407
─────────────────────────────────────────────

Volatility parameters:
──────────────────────────────────────────────
      Estimate   Std.Error   z value  Pr(>|z|)
──────────────────────────────────────────────
ω   0.00232225  0.00163909   1.41679    0.1565
β₁  0.884488    0.036963    23.929      <1e-99
α₁  0.124866    0.0405471    3.07952    0.0021
──────────────────────────────────────────────

Distribution parameters:
─────────────────────────────────────────
   Estimate  Std.Error  z value  Pr(>|z|)
─────────────────────────────────────────
ν   4.11211   0.400384  10.2704    <1e-24
─────────────────────────────────────────
```

It should, however, rarely be necessary to construct a `UnivariateARCHModel` manually via its constructor; typically, instances of it are created by calling [`fit`](@ref), [`selectmodel`](@ref), or [`simulate`](@ref).

!!! note
    If you *do* manually construct a `UnivariateARCHModel`, be aware that the constructor does not create copies of its arguments. This means that, e.g., calling `simulate!` on the constructed model will modify your data vector:
    ```jldoctest TYPES
    julia> mydata = copy(BG96); mydata[end]
    0.528047

    julia> am = UnivariateARCHModel(ARCH{0}([1.]), mydata);

    julia> simulate!(am);

    julia> mydata[end] ≈ 0.528047
    false
    ```
As discussed earlier, [`UnivariateARCHModel`](@ref) implements the interface of `StatisticalModel` from [`StatsBase`](http://juliastats.github.io/StatsBase.jl/stable/statmodels.html), so you
can call `coef`, `coefnames`, `confint`, `dof`, `informationmatrix`, `isfitted`, `loglikelihood`, `nobs`,  `score`, `stderror`, `vcov`, etc. on its instances:

```jldoctest TYPES
julia> nobs(am)
1974
```

Other useful methods include [`means`](@ref), [`volatilities`](@ref) and [`residuals`](@ref).
