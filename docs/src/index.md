```@meta
CurrentModule = NTRsurv
CollapsedDocStrings = true
```

# NTRsurv.jl

This package provides a Bayesian nonparametric workflow for survival analysis using neutral-to-the-right (NTR) priors, yielding principled and computationally efficient alternatives to classical frequentist Kaplan–Meier and Cox regression methods, with the possibility of incorporating prior information. The package implements prior and posterior simulation of survival curves, analytic posterior mean computation and Monte-Carlo credible
bands, allowing Cox-type regression modeling.

## Installation

`NTRsurv.jl` is registered in Julia's General registry and can be
installed using Julia's package manager:
```julia
julia> using Pkg
julia> Pkg.add("NTRsurv")
```