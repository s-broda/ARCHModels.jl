using Documenter, ARCH
makedocs(modules=[ARCH],
        doctest=true)

deploydocs(deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo = "https://github.com/s-broda/ARCH.jl.git",
    julia  = "0.6",
    osname = "linux")
