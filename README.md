# NTRsurv.jl

[![CI](https://github.com/alan7riva/NTRsurv.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/alan7riva/NTRsurv.jl/actions/workflows/CI.yml)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://alan7riva.github.io/NTRsurv.jl/stable/)


A Julia package for Bayesian nonparametric survival analysis with Neutral-to-the-Right (NTR) priors.

It provides tools for:
- Posterior mean estimation of survival function.
- Posterior simulation of survival functions.
- Credible bands for survival curves.

## Installation

`NTRsurv.jl` is registered in Julia's General registry:

    using Pkg
    Pkg.add("NTRsurv")

## Documentation

[Stable documentation](https://alan7riva.github.io/NTRsurv.jl/stable/)

## Further examples

The [`examples/`](examples/) directory contains simulation studies and
real-data analyses illustrating:

- Weibull simulation study containing theoretical details of the model;
- prior calibrations for small samples;
- Cox-NTR regression with uncertainty quantification focus;
- Meta-analysis example.

## License

`NTRsurv.jl` is distributed under the MIT License.