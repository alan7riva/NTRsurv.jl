```@meta
CurrentModule = NTRsurv
```

# NTRsurv.jl

This package provides a Bayesian nonparametric workflow for survival analysis using neutral-to-the-right (NTR) priors, yielding principled and computationally efficient alternatives to classical frequentist Kaplan–Meier and Cox regression methods, with the possibility of incorporating prior information.

## Installation

The package is currently available on GitHub and can be installed using Julia’s package manager:

```julia
using Pkg
Pkg.add(url="https://github.com/alan7riva/NTRsurv.jl.git")
using NTRsurv
```

## Getting started

## Dataset

To demonstrate the use of the package on real data, we consider a classical dataset from a clinical trial conducted by the North Central Cancer Treatment Group and described in Loprinzi et al. (1994). The data consists of 228 patients with advanced lung cancer, with survival time measured in days from enrollment and right censoring present. The dataset is widely used in introductory survival analysis tutorials and is included in the R `survival` package.

The dataset is distributed with the package as a CSV file in the `test/data/`
directory and is loaded as a `DataFrame` using the
[CSV](https://github.com/JuliaData/CSV.jl) and
[DataFrames](https://github.com/JuliaData/DataFrames.jl) packages.

```julia-repl
julia> using NTRsurv, Distributions, CSV, DataFrames

julia> lung = CSV.read(joinpath(pkgdir(NTRsurv), "test", "data", "lung.csv"), DataFrame);
```

## Fitting neutral to the right (NTR) model

```julia-repl
julia> T = lung[!,:time]
julia> δ = lung[!,:status]
```

```julia
using NTRsurv
using Distributions

T = rand(Weibull(2.5, 0.7), 200)
δ = ones(Int, length(T))

data = SurvivalData(T, δ)
baseline = ExponentialBaseline(1.0)
model = NeutralToTheRightModel(5.0, baseline, data)

t = collect(range(0.0, maximum(T), length = 100))

Smean = mean_posterior_survival(t, model)
lower, center, upper = posterior_credible_band(0.05, 1000, t, model)
```
## Contents

```@contents
Pages = [
    "events.md",
    "km.md",
    "na.md",
    "cox.md",
]
Depth = 1
```