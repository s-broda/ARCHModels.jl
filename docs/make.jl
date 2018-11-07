push!(LOAD_PATH,"../src/")
using Documenter, ARCH, Pkg
makedocs(modules=[ARCH],
        doctest=true
        strict=true)

deploydocs(repo = "github.com/s-broda/ARCH.jl.git")
