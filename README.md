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


- [Weibull simulation study](tutorials/Weibull_simulation_study.html)([source notebook](https://github.com/alan7riva/NTRsurv.jl/blob/main/examples/Weibull_simulation_study.ipynb))
    Simulation study including theoretical details for neutral to the right priors and illustrating their asymptotic properties.  

- [Small-sample survival analysis](tutorials/small_sample_analysis.html)([source notebook](https://github.com/alan7riva/NTRsurv.jl/blob/main/examples/small_sample_analysis.ipynb)) 
    Real data study illustrating informative and weakly informative baseline prior specifications effect on small samples, with an emphasis on tail behavior of the posterior.  

- [Uncertainty quantification for Cox regression models](tutorials/Cox_NTR_UQ.html)([source notebook](https://github.com/alan7riva/NTRsurv.jl/blob/main/examples/Cox_NTR_UQ.ipynb)) 
    Real data study illustrating NTR-Cox regression modelling, uncertainty quantification with posterior credible bands compared with frequentist confidence bands and posterior restricted survival mean contrasts.  

- [Meta-analysis for cohort data](tutorials/meta_analysis.html)([source notebook](https://github.com/alan7riva/NTRsurv.jl/blob/main/examples/meta_analysis.ipynb)) 
    Meta-analysis with cohort frailties NTR-Cox regression models using a cohort depending Cox regression function.


## License

`NTRsurv.jl` is distributed under the MIT License.