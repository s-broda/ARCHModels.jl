[![Build status (Linux, OSX)](https://travis-ci.org/s-broda/ARCHModels.jl.svg?branch=master)](https://travis-ci.org/s-broda/ARCHModels.jl) [![Build status (Windows)](https://ci.appveyor.com/api/projects/status/9ys3go5ng9j2jin5/branch/master?svg=true)](https://ci.appveyor.com/project/s-broda/archmodels-jl/branch/master) [![Coverage (coveralls)](https://coveralls.io/repos/s-broda/ARCHModels.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/s-broda/ARCHModels.jl?branch=master) [![Coverage (codecov)](http://codecov.io/github/s-broda/ARCHModels.jl/coverage.svg?branch=master)](http://codecov.io/github/s-broda/ARCHModels.jl?branch=master) [![Docs (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://s-broda.github.io/ARCHModels.jl/stable) [![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://s-broda.github.io/ARCHModels.jl/dev)

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

# Acknowledgements

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 750559).

<img src="docs/src/assets/LOGO.jpg" width="240">
