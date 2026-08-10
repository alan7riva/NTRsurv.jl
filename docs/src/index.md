```@meta
CurrentModule = NTRsurv
CollapsedDocStrings = true
```

# NTRsurv.jl

This package provides a Bayesian nonparametric workflow for survival analysis using neutral-to-the-right (NTR) priors, yielding principled and computationally efficient alternatives to classical frequentist Kaplan–Meier and Cox regression methods, with the possibility of incorporating prior information. The package implements prior and posterior simulation of survival curves, analytic posterior mean computation and Monte-Carlo credible, allowing Cox-type regression modeling.

## Installation

The package is currently available on GitHub and can be installed using Julia’s package manager:

```julia
julia> using Pkg
julia> Pkg.add("NTRsurv")
```

## Quick start: the Rossi recidivism data

We illustrate the workflow with the Rossi recidivism data. The event-time data contain observed follow-up
times in weeks and indicators of whether a rearrest occurred during the follow-up period.

```@setup rossi
using Random
using CSV
using DataFrames
using Plots
using Survival
using StatsModels
using NTRsurv
using Statistics

Random.seed!(1234)

Rossi = CSV.read(
    joinpath(pkgdir(NTRsurv), "examples", "data", "rossi.csv"),
    DataFrame
)

T = Float64.(Rossi.week)
δ = Int64.(Rossi.arrest)

t = collect(range(0.0, maximum(T), length = 100));

z_1 = [ 0.0, 23.0, 1.0, 1.0, 0.0, 1.0, 2.0 ]
z_2 = [ 1.0, 23.0, 1.0, 1.0, 0.0, 1.0, 2.0 ]
z_v = [z_1,z_2]
```

The sample size, number of observed events, and number of right-censored
observations are:

```@repl rossi
(nrow(Rossi), sum(δ), length(δ) - sum(δ))
```

### Constructing the NTR model

A `SurvivalData` object stores the observed times, censoring indicators, and the
sufficient statistics required by the NTR posterior computations. The `baseline` 
determines the prior mean survival curve, while `α` modulates the prior dispersion.
Here we use an empirical Bayes baseline constructed from the data.


```@repl rossi
data = SurvivalData(T, δ);
baseline = EmpiricalBayesBaseline(data);
α = 2.0;
model = NeutralToTheRightModel(α, baseline, data);
```

### Posterior mean and credible band

The posterior mean is available analytically, while credible bands are computed
from posterior survival draws; both are evaluated on a time mesh `t`.

```@repl rossi
posterior_mean = mean_posterior_survival(t, model);
band_lower, band_center, band_upper = posterior_credible_band(0.05, 3000, t, model; μ=false);
```

The argument `p = 0.05`, for the credble band, discards five percent of the `l=3000` sampled paths in the
Monte-Carlo band computation, yielding an approximate 95% credible band. The returned 
middle curve is the Monte Carlo posterior mean by default, when argument `μ` is omitted, 
or the Monte Carlo posterior median when `μ = false`.

For comparison, we also compute the Kaplan--Meier estimator using
[`Survival.jl`](https://juliastats.org/Survival.jl/latest/).

```@repl rossi
km = fit(Survival.KaplanMeier, T, δ);
NTR_baseline = exp.(-baseline.κ.(t));
```


```@repl rossi
NTR_plot = plot( t, band_center, ribbon = ( band_center .- band_lower, band_upper .- band_center), fillalpha = 0.25, linewidth = 2.2, xlabel = "Time (weeks)", ylabel = "Survival", ylims = (0.0, 1.0), label = "NTR posterior mean with 95% credible band", title = "NTR survival analysis of the Rossi data", size = (720, 430))
plot!( NTR_plot, km.events.time, km.survival, seriestype = :steppost, linewidth = 2, linestyle = :dash, label = "Kaplan--Meier estimator")
plot!( NTR_plot, t, NTR_baseline, linewidth = 2, linestyle = :dot, label = "Prior mean survival (baseline)")
savefig(NTR_plot, "rossi_ntr_fit.svg"); nothing # hide
```

![](rossi_ntr_fit.svg)


## Plug-in Cox-NTR regression

We set a `RegressionSurvivalData` object which stores the observed times, censoring 
indicators, covariates and required sufficient statistics for the Cox-NTR posterior 
computations.

```@repl rossi
z_keys = [:fin, :age, :race, :wexp, :mar, :paro, :prio];
Z = [ [Float64(Rossi[j, key]) for key in z_keys] for j in 1:nrow(Rossi) ];
dataregre = RegressionSurvivalData(T, δ, Z);
```

A practical Cox-NTR workflow first obtains a point estimate of the regression
coefficients and then conditions the Cox-NTR posterior computations on that
estimate. For example, either estimates from an MCMC sampler of the posterior 
dstribution or using the frequentist estimator are good alternatives.

```@setup rossi
# Compilation runs
NTRsurv.loglikelihood([0.3,0.1,-0.3,0.3,0.1,-0.3,-0.1],α,baseline,dataregre) # In this notebbok loglikelihood funcion is also loaded by Survival so it has to be sepcified.
robbins_monro_mh_within_gibbs_tune(3,3,x->NTRsurv.loglikelihood(x,α,baseline,dataregre),zeros(7),0.1.*ones(7),0.4.*ones(7),0.7)
```

```@repl rossi
# Robbins-Monro algorithm for variance of proposal distribution tuning
sd_prop_tuned, s₀_tune, lliks₀_tune = robbins_monro_mh_within_gibbs_tune(150,100,x->NTRsurv.loglikelihood(x,α,baseline,dataregre),zeros(7),0.1.*ones(7),0.4.*ones(7),0.7;show_progress=false);
# Metropolis-Hastings within Gibbs chain run
chain_s, _ =  random_walk_mh_within_gibbs( 3000, x-> NTRsurv.loglikelihood(x,α,baseline,dataregre), s₀_tune, lliks₀_tune, sd_prop_tuned[end]);
# Posterior mean estimate of regression coefficients for plug-in NTR-Cox model
c_post = mean(chain_s)
```

We can see that the posterior mean estimator of the Bayesian NTR workflow 
is quite similar to the frequentist Cox partial likelihood estimator.

```@repl rossi
Rossi.event = Survival.EventTime.(Rossi.week, Rossi.arrest .== 1);
freq_Cox_fit = Survival.coxph( @formula(event ~ fin + age + race + wexp + mar + paro + prio), Rossi);
c_freq = coef(freq_Cox_fit);
```
We construct the posterior mean plug-in Cox-NTR model and evaluate credible bands for patients with 
covariate vectors `z_1 = [0.0, 23.0, 1.0, 1.0, 0.0, 1.0, 2.0]` and `z_2 = [1.0, 23.0, 1.0, 1.0, 0.0, 1.0, 2.0]`
corresponding to the median population with (`z_1[1]=1.0`) and without (`z_1[1]=0.0`) financial aid; both 
covariate vectors are contained in `z_v = [z_1, z_2]`.

```@repl rossi
NTR_Cox_model = CoxNeutralToTheRightModel( c_post, α, baseline, dataregre);
NTR_Cox_bands = posterior_credible_band( 0.05, 3000, t, z_v, NTR_Cox_model);
# Plot for financial aid vs no fiancial aid median survival curves
NTR_Cox_plot =  plot( t, NTR_Cox_bands[1][2],  ribbon = ( NTR_Cox_bands[1][2] .- NTR_Cox_bands[1][1], NTR_Cox_bands[1][3] .- NTR_Cox_bands[1][2]), c=4, xlabel="\$t\$", ylabel="\$S(t)\$", fillalpha=0.3, label="No financial aid." , title="Median survival curves")
plot!( t, NTR_Cox_bands[2][2],  ribbon = ( NTR_Cox_bands[2][2] .- NTR_Cox_bands[2][1], NTR_Cox_bands[2][3] .- NTR_Cox_bands[2][2]), c=5, fillalpha=0.3,  label="Financial aid", size = (720, 430))
savefig(NTR_Cox_plot, "rossi_cox_ntr_fit.svg"); nothing # hide
```

![](rossi_cox_ntr_fit.svg)


## Public interface

### Data and baseline specification

```@docs
SurvivalData
RegressionSurvivalData
Baseline
ExponentialBaseline
WeibullBaseline
EmpiricalBayesBaseline
```

```@docs
NeutralToTheRightModel
CoxNeutralToTheRightModel
 CoxNeutralToTheRightFullyBayesianModel
```

### Posterior summaries and simulation
```@docs
mean_posterior_survival
sample_prior_survival
sample_posterior_survival
credible_band
prior_credible_band
posterior_credible_band
```

### Likelihood and MCMC utilities

```@docs
loglikelihood
random_walk_mh
random_walk_mh_within_gibbs
robbins_monro_mh_tune
robbins_monro_mh_within_gibbs_tune
acceptance_rate
```

## Index

```@index
```