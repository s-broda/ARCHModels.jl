struct DCC{p, q, VS<:VolatilitySpec, T<:AbstractFloat, d} <: MultivariateVolatilitySpec{T, d}
    coefs::Vector{T}
    univariatespecs::Vector{VS}
    function DCC{p, q, VS, T, d}(coefs::Vector{T}, univariatespecs:: Array{VS}) where {p, q, VS<:VolatilitySpec, T, d}
        length(coefs) == nparams(DCC{p, q})  || throw(NumParamError(nparams(DCC{p, q}), length(coefs)))
        @assert d == length(univariatespecs)
        new{p, q, VS, T, d}(coefs, univariatespecs)
    end
end
DCC{p, q}(coefs::Vector{T}, univariatespecs::Array{VS}) where {p, q, T, VS<:VolatilitySpec{T}} = DCC{p, q, VS, T, length(univariatespecs)}(coefs, univariatespecs)

nparams(::Type{DCC{p, q}}) where {p, q} = p+q

function fit(::Type{<:DCC{p, q, VS}}, data::Matrix{T}) where {p, q, VS<: VolatilitySpec, T, d}
    for i = 1:size(data, 2)
        m = fit(VS, data[:, i])
        println(m.spec.coefs)
    end
end
