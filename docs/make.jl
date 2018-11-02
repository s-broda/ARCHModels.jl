push!(LOAD_PATH,"../src/")
using Documenter, ARCH
makedocs(modules=[ARCH],
        doctest=true)

deploydocs(repo = "github.com/s-broda/ARCH.jl.git")
