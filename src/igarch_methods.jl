# IGARCH has infinite unconditional variance. The generic _simulate! would push
# Inf into the presample (Inf > 0 is true). Start from a finite value instead.
function _simulate!(data::Vector{T2}, spec::IGARCH{p, q, T2};
                  warmup=100,
                  dist::StandardizedDistribution{T2}=StdNormal{T2}(),
                  meanspec::MeanSpec{T2}=NoIntercept{T2}(),
                  rng=GLOBAL_RNG
                  ) where {p, q, T2<:AbstractFloat}
    @assert warmup>=0
    append!(data, zeros(T2, warmup))
    Tlen = length(data)
    r1 = presample(typeof(spec))
    r2 = presample(meanspec)
    r = max(max(r1, r2), 1)
    ht = CircularBuffer{T2}(r)
    lht = CircularBuffer{T2}(r)
    zt = CircularBuffer{T2}(r)
    at = CircularBuffer{T2}(r)
    @inbounds begin
        h0 = one(T2)
        m0 = uncond(meanspec)
        h0 > 0 || error("Model is nonstationary.")
        for t = 1:Tlen
            if t>r2
                themean = mean(at, ht, lht, data, meanspec, meanspec.coefs, t)
            else
                themean = m0
            end
            if t>r1
                update!(ht, lht, zt, at, typeof(spec), spec.coefs)
            else
                push!(ht, h0)
                push!(lht, log(h0))
            end
            push!(zt, rand(rng, dist))
            push!(at, sqrt(ht[end])*zt[end])
            data[t] = themean + at[end]
        end
    end
    deleteat!(data, 1:warmup)
end

# Multi-step variance is defined for IGARCH (forecasts rise linearly in ω).
# The generic predict method only allows this for TGARCH via `VS <: TGARCH`.
function predict(am::UnivariateARCHModel{T, VS, SD}, what=:volatility, horizon=1; level=0.01) where {T, VS<:IGARCH, SD}
    ht = volatilities(am).^2
    lht = log.(ht)
    zt = residuals(am)
    at = residuals(am, standardized=false)
    themean = T(0)
    if horizon > 1
        if what == :VaR
            error("Predicting VaR more than one period ahead is not implemented. Consider predicting one period ahead and scaling by `sqrt(horizon)`.")
        elseif what == :volatility
            error("Predicting volatility more than one period ahead is not implemented.")
        end
    end
    data = copy(am.data)
    for current_horizon = (1 : horizon)
        t = length(am.data) + current_horizon
        if what == :return || what == :VaR
            themean = mean(at, ht, lht, data, am.meanspec, am.meanspec.coefs, t)
        end
        update!(ht, lht, zt, at, VS, am.spec.coefs, current_horizon)
        push!(zt, 0.)
        push!(at, 0.)
        push!(data, themean)
    end
    if what == :return
        return themean
    elseif what == :volatility
        return sqrt(ht[end])
    elseif what == :variance
        return ht[end]
    elseif what == :VaR
        return -themean - sqrt(ht[end]) * quantile(am.dist, level)
    else
        error("Prediction target $(what) unknown.")
    end
end
