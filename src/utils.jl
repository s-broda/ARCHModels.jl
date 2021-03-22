const MatOrVec{T} = Union{Matrix{T}, Vector{T}} where T

Base.@irrational sqrt2invpi 0.79788456080286535587 sqrt(big(2)/big(π))

#from here https://stackoverflow.com/questions/46671965/printing-variable-subscripts-in-julia
subscript(i::Integer) = i<0 ? error("$i is negative") : join('₀'+d for d in reverse(digits(i)))

#count the number of type vars. there's probably a better way.
function my_unwrap_unionall(@nospecialize a)
    count = 0
    while isa(a, UnionAll)
        a = a.body
        count += 1
    end
    return count
end

@inline function to_corr(Σ)
	D = sqrt(abs.(Diagonal(Σ))) # horrible hack. required to fix a non-deterministic doctest failure
    iD = inv(D)
    R = iD * Σ * iD
    R = (R + R') / 2
end

#=
    analytical_shrinkage(X::Matrix)
Analytical nonlinear shrinkage estimator of the covariance matrix. Based on the
Matlab code from [1]. Translated to Julia and used here under MIT license by
permission from the authors.

[1] Ledoit, O., and Wolf, M. (2018), "Analytical Nonlinear Shrinkage of
Large-Dimensional Covariance Matrices", University of Zurich Econ WP 264.
https://www.econ.uzh.ch/static/workingpapers_iframe.php?id=943
=#
function analytical_shrinkage(X)
n, p = size(X)
@assert n >= 12 # important: sample size n must be >= 12
sample = Symmetric(X'*X) / n
E = eigen(sample)
lambda = E.values
u = E.vectors

# compute analytical nonlinear shrinkage kernel formula
lambda = lambda[max(1, p-n+1):p]
L = repeat(lambda, 1, min(p, n))
h = n^(-1/3) # Equation (4.9)
H = h*L'
x = (L-L') ./ H
ftilde = (3/4/sqrt(5)) * mean(max.(1 .- x.^2 ./ 5, 0) ./ H, dims=2) # Equation (4.7)
Hftemp = (-3/10/pi) * x + (3/4/sqrt(5)/pi) * (1 .- x.^2 ./ 5) .* log.(abs.((sqrt(5).-x) ./ (sqrt(5).+x))) # Equation (4.8)
Hftemp[abs.(x) .== sqrt(5)] .= (-3/10/pi) .* x[abs.(x) .== sqrt(5)]
Hftilde = mean(Hftemp./H, dims=2)
if p<=n
    dtilde = lambda ./ ((pi * (p/n) *lambda .* ftilde).^2  + (1 .- (p/n) .- pi * (p/n) * lambda .* Hftilde).^2) # Equation (4.3)
else
    Hftilde0 = (1/pi) * (3/10/h^2 + 3/4/sqrt(5)/h*(1-1/5/h^2) * log((1+sqrt(5)*h)/(1-sqrt(5)*h)))*mean(1 ./ lambda) # Equation (C.8)
    dtilde0 = 1 / (pi * (p-n) / n * Hftilde0) # Equation (C.5)
    dtilde1 = lambda ./ (pi^2*lambda.^2 .* (ftilde.^2 + Hftilde.^2)) # Eq. (C.4)
    dtilde = [dtilde0*ones(p-n,1); dtilde1]
end
return u * Diagonal(dtilde[:]) * u'
end
