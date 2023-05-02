[![version](https://juliahub.com/docs/ARCHModels/version.svg)](https://juliahub.com/ui/Packages/ARCHModels/cpjxl) [![Docs (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://s-broda.github.io/ARCHModels.jl/stable) [![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://s-broda.github.io/ARCHModels.jl/dev) [![Build status (Linux, MacOS)](https://github.com/s-broda/ARCHModels.jl/workflows/CI/badge.svg)](https://github.com/s-broda/ARCHModels.jl/actions?query=workflow%3ACI) [![Build status (Windows)](https://ci.appveyor.com/api/projects/status/9ys3go5ng9j2jin5/branch/master?svg=true)](https://ci.appveyor.com/project/s-broda/archmodels-jl/branch/master) [![Coverage (codecov)](http://codecov.io/github/s-broda/ARCHModels.jl/coverage.svg?branch=master)](http://codecov.io/github/s-broda/ARCHModels.jl?branch=master) [![pkgeval](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/A/ARCHModels.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/A/ARCHModels.html) [![DOI](https://zenodo.org/badge/95967480.svg)](https://zenodo.org/badge/latestdoi/95967480)
# The ARCHModels Package for Julia

ARCH (Autoregressive Conditional Heteroskedasticity) models are a class of models designed to capture a feature of financial returns data known as *volatility clustering*, *i.e.*, the fact that large (in absolute value) returns tend to cluster together, such as during periods of financial turmoil, which then alternate with relatively calmer periods. This package provides efficient routines for simulating, estimating, and testing a variety of GARCH models.

# Installation

`ARCHModels` is a registered Julia package. To install it in Julia 1.0 or later, do

```
add ARCHModels
```

in the Pkg REPL mode (which is entered by pressing `]` at the prompt).
# Documentation

The extensive documentation is available [here](https://s-broda.github.io/ARCHModels.jl/stable/).

# Citation
If you use this package in your research, please consider citing [our paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3551503).

# Acknowledgements

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 750559.

<img src="docs/src/assets/EULOGO.jpg" width="240">
