push!(LOAD_PATH,"../src/")
using Documenter, ARCHModels
DocMeta.setdocmeta!(ARCHModels, :DocTestSetup, :(using ARCHModels; using Random; Random.seed!(1)); recursive=true)
makedocs(modules=[ARCHModels],
        sitename="ARCHModels.jl",
        format = Documenter.HTML(assets=["assets/invenia.css"]),
        doctest=true,
        strict=true,
        pages = ["Home" => "index.md",
                 "Univariate ARCH Models" => Any["univariateintro.md", "univariateusage.md"],
                 "Multivariate ARCH Models" => Any["multivariateintro.md", "multivariateusage.md"],
                 "reference.md"
                 ]
        )

deploydocs(repo="github.com/s-broda/ARCHModels.jl.git")
