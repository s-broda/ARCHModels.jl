export GARCH
export _ARCH #ARCH conflicts with module name
struct GARCH{p, q, T<:AbstractFloat} <: VolatilitySpec{T}
    coefs::Vector{T}
    function GARCH{p, q, T}(coefs::Vector{T}) where {p, q, T}
        length(coefs) == nparams(GARCH{p, q})  || throw(NumParamError(nparams(GARCH{p, q}), length(coefs)))
        new{p, q, T}(coefs)
    end
end
GARCH{p, q}(coefs::Vector{T}) where {p, q, T}  = GARCH{p, q, T}(coefs)


const _ARCH = GARCH{0}

@inline nparams(::Type{<:GARCH{p, q}}) where {p, q} = p+q+1

@inline presample(::Type{<:GARCH{p, q}}) where {p, q} = max(p, q)

@inline function update!(ht, lht, zt, ::Type{<:GARCH{p,q}}, MS::Type{<:MeanSpec},
                         data, garchcoefs, meancoefs, t
                         ) where {p, q}
    ht[t] = garchcoefs[1]
    for i = 1:p
        ht[t] += garchcoefs[i+1]*ht[t-i]
    end
    for i = 1:q
        ht[t] += garchcoefs[i+1+p]*(data[t-i]-mean(MS, meancoefs))^2
    end
    lht[t] = log(ht[t])
    zt[t] = (data[t]-mean(MS, meancoefs))/sqrt(ht[t])
    return nothing
end

@inline function uncond(::Type{<:GARCH{p, q}}, coefs::Vector{T}) where {p, q, T}
    den=one(T)
    for i = 1:p+q
        den -= coefs[i+1]
    end
    h0 = coefs[1]/den
end

function startingvals(::Type{<:GARCH{p,q}}, data::Array{T}) where {p, q, T}
    x0 = zeros(T, p+q+1)
    x0[2:p+1] = 0.9/p
    x0[p+2:end] = 0.05/q
    x0[1] = var(data)*(one(T)-sum(x0))
    return x0
end

function constraints(::Type{<:GARCH{p,q}}, ::Type{T}) where {p, q, T}
    lower = zeros(T, p+q+1)
    upper = ones(T, p+q+1)
    upper[1] = T(Inf)
    return lower, upper
end

function coefnames(::Type{<:GARCH{p,q}}) where {p, q}
    names = Array{String, 1}(p+q+1)
    names[1] = "ω"
    names[2:p+1] .= (i -> "β"*subscript(i)).([1:p...])
    names[p+2:p+q+1] .= (i -> "α"*subscript(i)).([1:q...])
    return names
end
