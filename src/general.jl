"""
	ARCHModel <: StatisticalModel
"""
abstract type ARCHModel <: StatisticalModel end

"""
	VolatilitySpec{T}
Abstract supertype of UnivariateVolatilitySpec{T} and MultivariateVolatilitySpec{T} .
"""
abstract type VolatilitySpec{T} end

"""
    MeanSpec{T}
Abstract supertype that mean specifications inherit from.
"""
abstract type MeanSpec{T} end

struct NumParamError <: Exception
    expected::Int
    got::Int
end

function showerror(io::IO, e::NumParamError)
    print(io, "incorrect number of parameters: expected $(e.expected), got $(e.got).")
end

nobs(am::ARCHModel) = length(am.data)
islinear(am::ARCHModel) = false
isfitted(am::ARCHModel) = am.fitted

function confint(am::ARCHModel, level::Real=0.95)
    hcat(coef(am), coef(am)) .+ stderror(am)*quantile(Normal(),(1. -level)/2.)*[1. -1.]
end

score(am::ARCHModel) = sum(scores(am), dims=1)

function vcov(am::ARCHModel)
	S = scores(am)
	V = S'S/nobs(am)
    J = informationmatrix(am; expected=false) #Note: B&W use expected information.
    Ji = try
        inv(J)
    catch e
        if e isa LinearAlgebra.SingularException
            @warn "Fisher information is singular; vcov matrix is inaccurate."
            pinv(J)
        else
            rethrow(e)
        end
    end
    v = Ji*V*Ji/nobs(am) #Huber sandwich
    all(diag(v).>0) || @warn "non-positive variance encountered; vcov matrix is inaccurate."
    v
end

function show(io::IO, spec::VolatilitySpec)
    println(io, modname(typeof(spec)), " specification.\n\n", length(spec.coefs) > 0 ? CoefTable(spec.coefs, coefnames(typeof(spec)), ["Parameters:"]) : "No estimable parameters.")
end

stderror(am::ARCHModel) = sqrt.(abs.(diag(vcov(am))))

"""
    fit!(am::ARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)

Fit the uni- or multivariate ARCHModel specified by `am`, modifying `am` in place.
Keyword arguments are passed on to the optimizer.
"""
function fit!(am::ARCHModel; kwargs...) end

"""
    fit(am::ARCHModel; algorithm=BFGS(), autodiff=:forward, kwargs...)

Fit the uni- or multivariate ARCHModel specified by `am` and return the result in a new instance of
`ARCHModel`. Keyword arguments are passed on to the optimizer.
"""
function fit(am::ARCHModel; kwargs...) end

"""
    simulate!(am::ARCHModel; warmup=100)
Simulate an ARCHModel, modifying `am` in place.
"""
function simulate! end

"""
    simulate(am::ARCHModel; warmup=100)
	simulate(am::ARCHModel, T; warmup=100)
    simulate(spec::UnivariateVolatilitySpec, T; warmup=100, dist=StdNormal(), meanspec=NoIntercept())
Simulate a length-T time series from a UnivariateARCHModel.
	simulate(spec::MultivariateVolatilitySpec, T; warmup=100, dist=MultivariateStdNormal(), meanspec=[NoIntercept() for i = 1:d])
Simulate a length-T time series from a MultivariateARCHModel.
"""
function simulate end

function simulate!(am::ARCHModel; warmup=100)
	am.fitted = false
    _simulate!(am.data, am.spec; warmup=warmup, dist=am.dist, meanspec=am.meanspec)
    am
end

function simulate(am::ARCHModel, nobs; warmup=100)
	am2 = deepcopy(am)
	simulate(am2.spec, nobs; warmup=warmup, dist=am2.dist, meanspec=am2.meanspec)
end

simulate(am::ARCHModel; warmup=100) = simulate(am, size(am.data)[1]; warmup=warmup)
