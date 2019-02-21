export ARCHLMTest, DQTest

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
    ARCHLMTest(am::UnivariateARCHModel, p=max(o, p, q, ...))
Conduct Engle's (1982) LM test for autoregressive conditional heteroskedasticity with
p lags in the test regression.
"""
ARCHLMTest(am::UnivariateARCHModel, p=presample(typeof(am.spec))) = ARCHLMTest(residuals(am), p)

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
population_param_of_interest(x::ARCHLMTest) = ("T⋅R² in auxiliary regression of rₜ² on an intercept and its own lags", 0, x.LM)
function show_params(io::IO, x::ARCHLMTest, ident)
    println(io, ident, "sample size:                    ", x.n)
    println(io, ident, "number of lags:                 ", x.p)
    println(io, ident, "LM statistic:                   ", x.LM)
end

pvalue(x::ARCHLMTest) = pvalue(Chisq(x.p), x.LM; tail=:right)


"""
    DQTest <: HypothesisTest
Engle and Manganelli's (2004) out-of-sample dynamic quantile test.
"""
struct DQTest{T<:Real} <: HypothesisTest
    n::Int         # number of observations
    p::Int         # number of lags
    level::T       # VaR level
    DQ::T          # test statistic
end

"""
    DQTest(data, vars, level, p=1)
Conduct Engle and Manganelli's (2004) out-of-sample dynamic quantile test with
p lags in the test regression. `vars` shoud be a vector of out-of-sample Value at Risk
predictions at level `level`.
"""
function DQTest(data::Vector{T}, vars::Vector{T}, level::AbstractFloat, p::Integer=1) where T<:Real
    @assert p>0
    @assert length(data) == length(vars)
    n = length(data)
    hit = (data .< -vars).*1 .- level
    y = hit[p+1:n]
    X = zeros(T, (n-p, p+2))
    X[:, 1] .= one(T)
    for i in 1:p
        X[:, i+1] = hit[p-i+1:n-i]
    end
    X[:, p+2] = vars[p+1:n]
    B = X \ y
    DQ = B' * (X'*X) *B/(level*(1-level)) # y'X * inv(X'X) * X'y / (level*(1-level)); note 2 typos in the paper
    DQTest(n, p, level, DQ)
end

testname(::DQTest) = "Engle and Manganelli's (2004) DQ test (out of sample)"
population_param_of_interest(x::DQTest) = ("Wald statistic in auxiliary regression", 0, x.DQ)
function show_params(io::IO, x::DQTest, ident)
    println(io, ident, "sample size:                    ", x.n)
    println(io, ident, "number of lags:                 ", x.p)
    println(io, ident, "VaR level:                      ", x.level)
    println(io, ident, "DQ statistic:                   ", x.DQ)
end

pvalue(x::DQTest) = pvalue(Chisq(x.p+2), x.DQ; tail=:right)
