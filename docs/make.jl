push!(LOAD_PATH,"../src/")
using Documenter, ARCH, Pkg
makedocs(modules=[ARCH],
        sitename="ARCH.jl Documentation",
        doctest=true,
        strict=true)

deploydocs(repo="github.com/s-broda/ARCH.jl.git",
           julia="1.0")
