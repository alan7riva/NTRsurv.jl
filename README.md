# NTRsurv.jl

[![CI](https://github.com/alan7riva/NTRsurv.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/alan7riva/NTRsurv.jl/actions/workflows/CI.yml)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://alan7riva.github.io/NTRsurv.jl/stable/)


A Julia package for Bayesian nonparametric survival analysis with Neutral-to-the-Right (NTR) priors.

It provides tools for:
- Posterior mean estimation of survival function.
- Posterior simulation of survival functions.
- Credible bands for survival curves.
- Cox regression type modeling.

## Installation

`NTRsurv.jl` is registered in Julia's General registry:

    using Pkg
    Pkg.add("NTRsurv")

## Documentation

[Stable documentation](https://alan7riva.github.io/NTRsurv.jl/stable/)

## Further examples

The [`examples/`](examples/) directory contains simulation studies and
real-data analyses illustrating:

- [Weibull simulation study containing theoretical details of the model](https://nbviewer.org/github/alan7riva/NTRsurv.jl/blob/main/examples/Weibull_simulation_study.ipynb);
- [prior calibrations for small samples](https://nbviewer.org/github/alan7riva/NTRsurv.jl/blob/main/examples/small_sample_analysis.ipynb);
- [Cox-NTR regression with uncertainty quantification focus](https://nbviewer.org/github/alan7riva/NTRsurv.jl/blob/main/examples/Cox_NTR_UQ.ipynb);
- [Meta-analysis example](https://nbviewer.org/github/alan7riva/NTRsurv.jl/blob/main/examples/meta_analysis.ipynb).

## License

`NTRsurv.jl` is distributed under the MIT License.