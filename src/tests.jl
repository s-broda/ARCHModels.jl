export ARCHLMTest

"""
    ARCHLMTest <: HypothesisTest
Engle's (1982) LM test for autoregressive conditional heteroskedasticity.
"""
struct ARCHLMTest{T<:Real} <: HypothesisTest
    n::Int         # number of observations
    p::Int         # number of lags
    LM::T    # test statistic
end

"""
    ARCHLMTest(am::ARCHModel, p=max(o, p, q, ...))
Conduct Engle's (1982) LM test for autoregressive conditional heteroskedasticity with
p lags in the test regression.
"""
ARCHLMTest(am::ARCHModel, p=presample(typeof(am.spec))) = ARCHLMTest(residuals(am), p)

"""
    ARCHLMTest(u::Vector, p::Integer)
Conduct Engle's (1982) LM test for autoregressive conditional heteroskedasticity with
p lags in the test regression.
"""
function ARCHLMTest(u::Vector{T}, p::Integer) where T<:Real
    @assert p>0
    n = length(u)
    u2 = u.^2
    X = zeros(T, (n-p, p+1))
    X[:, 1] .= one(eltype(u))
    for i in 1:p
        X[:, i+1] = u2[p-i+1:n-i]
    end
    y = u2[p+1:n]
    B = X \ y
    e = y - X*B
    ybar = y .- mean(y)
    LM = n * (1 - (e'e)/(ybar'ybar)) #T*R^2
    ARCHLMTest(n, p, LM)
end

testname(::ARCHLMTest) = "ARCH LM test for conditional heteroskedasticity"
population_param_of_interest(x::ARCHLMTest) = ("T⋅R² in auxiliary regression of uₜ² on an intercept and its own lags", 0, x.LM)
function show_params(io::IO, x::ARCHLMTest, ident)
    println(io, ident, "sample size:                    ", x.n)
    println(io, ident, "number of lags:                 ", x.p)
    println(io, ident, "LM statistic:                   ", x.LM)
end

pvalue(x::ARCHLMTest) = pvalue(Chisq(x.p), x.LM; tail=:right)
