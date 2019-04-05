push!(LOAD_PATH,"../src/")
using Documenter, ARCHModels
makedocs(modules=[ARCHModels],
        sitename="ARCHModels.jl",
        format = Documenter.HTML(assets=["assets/invenia.css"]),
        doctest=true,
        strict=true,
        pages = ["Home" => "index.md",
                 "types.md",
                 "manual.md",
                 "reference.md"
                 ]
        )

deploydocs(repo="github.com/s-broda/ARCHModels.jl.git")
