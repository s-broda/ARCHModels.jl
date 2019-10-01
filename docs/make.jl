push!(LOAD_PATH,"../src/")
using Documenter, ARCHModels
DocMeta.setdocmeta!(ARCHModels, :DocTestSetup, :(using ARCHModels; using Random; Random.seed!(1)); recursive=true)
makedocs(modules=[ARCHModels],
        sitename="ARCHModels.jl",
        format = Documenter.HTML(assets=["assets/invenia.css"]),
        doctest=true,
        strict=true,
        pages = ["Home" => "index.md",
                 "introduction.md",
                 "Type Hierarchy" => Any[
                        "univariatetypehierarchy.md",
                        "multivariatetypehierarchy.md"
                        ],
                 "usage.md",
                 "reference.md"
                 ]
        )

deploydocs(repo="github.com/s-broda/ARCHModels.jl.git")
