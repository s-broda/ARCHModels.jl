```@meta
DocTestSetup = quote
    using Random
    Random.seed!(1)
end
DocTestFilters = r".*[0-9\.]"
```

We will be using the data from [Bollerslev and Ghysels](https://doi.org/10.2307/1392425), available as the constant [`BG96`](@ref). The data consist of daily German mark/British pound exchange rates (1974 observations) and are often used in evaluating
implementations of (G)ARCH models (see, e.g., [Brooks et.al.](https://doi.org/10.1016/S0169-2070(00)00070-4). We begin by convincing ourselves that the data exhibit ARCH effects; a quick and dirty way of doing this is to look at the sample autocorrelation function of the squared returns:

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

It should rarely be necessary to call the constructor directly; typically, instances of [`ARCHModel`](@ref) are created by calling [`simulate`](@ref) or [`fit`](@ref).

# Estimation

# Simulation
# Model selection
```@meta
DocTestSetup = nothing
DocTestFilters = nothing
```
