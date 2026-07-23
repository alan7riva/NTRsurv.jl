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
```

## Getting started

## Dataset

To demonstrate the use of the package on real data, we consider a classical dataset from a clinical trial conducted by the North Central Cancer Treatment Group and described in Loprinzi et al. (1994). The data consists of 228 patients with advanced lung cancer, with survival time measured in days from enrollment and right censoring present. The dataset is widely used in introductory survival analysis tutorials and is included in the R `survival` package.

The dataset is distributed with the package as a CSV file in the `test/data/`
directory and is loaded as a `DataFrame` using the
[CSV](https://github.com/JuliaData/CSV.jl) and
[DataFrames](https://github.com/JuliaData/DataFrames.jl) packages.


## Fitting neutral to the right (NTR) model

```@example ntr-fit
using NTRsurv, Distributions, CSV, DataFrames, Plots
lung = CSV.read(joinpath(pkgdir(NTRsurv), "test", "data", "lung.csv"), DataFrame);
```

### Baseline specification and prior inspection

The baseline cumulative hazard determines the prior mean survival curve. Here we use an empirical Bayes baseline constructed from the data.

```@example ntr-fit
T = lung[!,:time]
δ = lung[!,:status]
data = SurvivalData(T,δ)
baseline = EmpiricalBayesBaseline(data)
α = 2.0
t = collect(range(0.0, maximum(T), length = 100))
prior_band_d, prior_band_m, prior_band_u = prior_credible_band(0.05, 500, t, α, baseline)
prior_plot = plot(t,prior_band_m,ribbon = (prior_band_m .- prior_band_d,prior_band_u .- prior_band_m),fillalpha = 0.3,xlabel = "\$t\$",ylabel = "\$S_0(t)\$", label = "Prior mean baseline\nwith 95% band", title = "Empirical Bayes prior", size = (600, 400)) #hide
savefig(prior_plot, "ntr_prior_band.svg"); nothing # hide
```

![](ntr_prior_band.svg)


```@example ntr-fit
model = NeutralToTheRightModel( α, baseline, data)
NTR_band_d, NTR_band_m, NTR_band_u = posterior_credible_band(0.05,3000,t,model)
```

## Cox-NTR workflow

```julia
Z = [[randn()] for _ in eachindex(T)]

datareg = RegressionSurvivalData(T, δ, Z)
coxmodel = CoxNeutralToTheRightModel([0.5], 5.0, baseline, datareg)

znew = [0.2]

Smean_z = mean_posterior_survival(t, znew, coxmodel)
lower_z, center_z, upper_z = posterior_credible_band(0.05, 1000, t, znew, coxmodel)
```

## Credible bands

The argument `p` in `posterior_credible_band(p, ...)` is the fraction of posterior paths discarded to form the envelope. Thus, `p = 0.05` gives an approximate 95% posterior credible band. The returned middle curve is the Monte Carlo posterior mean by default, or the Monte Carlo posterior median when `mu = false`.

## Contents

```@contents
Pages = [
    "index.md",
    "api.md",
]
Depth = 2
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