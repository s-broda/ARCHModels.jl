push!(LOAD_PATH,"../src/")
using Documenter, ARCH
makedocs(modules=[ARCH],
        doctest=true)

deploydocs(deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo = "github.com/s-broda/ARCH.jl.git",
    julia  = "0.6",
    osname = "linux")
