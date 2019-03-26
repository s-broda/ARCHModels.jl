const MatOrVec{T} = Union{Matrix{T}, Vector{T}} where T

struct NumParamError <: Exception
    expected::Int
    got::Int
end

function showerror(io::IO, e::NumParamError)
    print(io, "incorrect number of parameters: expected $(e.expected), got $(e.got).")
end

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
