# ARCH

[![Build status (Linux, OSX)](https://travis-ci.org/s-broda/ARCH.jl.svg?branch=master)](https://travis-ci.org/s-broda/ARCH.jl) [![Build status (Windows)](https://ci.appveyor.com/api/projects/status/6b98se8nrsbl71nb/branch/master?svg=true)](https://ci.appveyor.com/project/s-broda/arch-jl/branch/master) [![Coverage (coveralls)](https://coveralls.io/repos/s-broda/ARCH.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/s-broda/ARCH.jl?branch=master) [![Coverage (codecov)](http://codecov.io/github/s-broda/ARCH.jl/coverage.svg?branch=master)](http://codecov.io/github/s-broda/ARCH.jl?branch=master) [![Docs (latest)](https://img.shields.io/badge/docs-latest-blue.svg)](https://s-broda.github.io/ARCH.jl/latest)

ARCH (Autoregressive Conditional Heteroskedasticity) models are a class of models designed to capture a feature of financial returns data known as *volatility clustering*, *i.e.*, the fact that large (in absolute value) returns tend to cluster together, such as during periods of financial turmoil, which then alternate with relatively calmer periods. This package provides efficient routines for simulating, estimating, and testing a variety of GARCH models.

# Installation

The package is not registered yet; to install it, do `Pkg.clone("https://github.com/s-broda/ARCH.jl")`. Only Julia 0.6 is supported at the moment.

# Acknowledgements

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 750559).

<img src="docs/src/assets/LOGO_ERC-FLAG_EU_.jpg" width="240">
