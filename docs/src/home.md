```@meta
CurrentModule = NTRsurv
```

# NTRsurv.jl

This package provides a Bayesian nonparametric workflow for survival analysis using neutral-to-the-right priors, yielding principled and computationally efficient alternatives to classical frequentist Kaplan–Meier and Cox regression methods, with the possibility of incorporating prior information.

## Installation

The package is currently available on GitHub and can be installed using Julia’s package manager:

`using Pkg; Pkg.add(url="https://github.com/alan7riva/NTRsurv.jl.git")`.

## Getting started

## Dataset

To demonstrate the use of the package on real data, we consider a classical dataset from a clinical trial conducted by the North Central Cancer Treatment Group and described in Loprinzi et al. (1994). The data consists of 228 patients with advanced lung cancer, with survival time measured in days from enrollment and right censoring present. The dataset is widely used in introductory survival analysis tutorials and is included in the R `survival` package.

The dataset is distributed with the package as a CSV file in the `test/data/`
directory and is loaded as a `DataFrame` using the
[CSV](https://github.com/JuliaData/CSV.jl) and
[DataFrames](https://github.com/JuliaData/DataFrames.jl) packages.

```julia-repl
julia> using NTRsurv, Survival, StatsModels, CSV, DataFrames

julia> lung = CSV.read(joinpath(pkgdir(NTRsurv), "test", "data", "lung.csv"), DataFrame);
```

## Fitting neutral to the right (NTR) model

```julia-repl
julia> T = lung[!,:time]
julia> δ = lung[!,:status]
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