push!(LOAD_PATH,"../src/")
using Documenter, ARCH
makedocs(modules=[ARCH],
        doctest=false)

deploydocs(deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo = "github.com/s-broda/ARCH.jl.git",
    julia  = "1.0",
    osname = "linux")
