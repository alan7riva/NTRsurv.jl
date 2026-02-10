```@meta
CurrentModule = Survival
```

# NTRsurv.jl

This package provides a Bayesian nonparametric workflow for survival analysis using neutral-to-the-right priors, yielding principled and computationally efficient alternatives to classical frequentist Kaplan–Meier and Cox regression methods, with the possibility of incorporating prior information.

## Installation

The package is currently available on GitHub and can be installed using Julia’s package manager:

`using Pkg; Pkg.add( https://github.com/alan7riva/NTRsurv.jl.git)`.

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