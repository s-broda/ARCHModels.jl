push!(LOAD_PATH,"../src/")
using Documenter, ARCH
makedocs(modules=[ARCH],
        sitename="ARCH.jl",
        assets=["assets/invenia.css"],
        doctest=true,
        strict=true,
        pages = ["Home" => "index.md",
                 "types.md",
                 "manual.md",
                 "reference.md"
                 ]
        )

deploydocs(repo="github.com/s-broda/ARCH.jl.git")
