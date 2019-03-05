"""
	ARCHModel <: StatisticalModel
"""
abstract type ARCHModel <: StatisticalModel end



"""
    MeanSpec{T}
Abstract supertype that mean specifications inherit from.
"""
abstract type MeanSpec{T} end


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

stderror(am::ARCHModel) = sqrt.(abs.(diag(vcov(am))))
