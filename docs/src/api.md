```@meta
CurrentModule = NTRsurv
CollapsedDocStrings = true
```

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
PluginCoxNeutralToTheRightModel
CoxNeutralToTheRightModel
```

### Posterior summaries and simulation
```@docs
mean_posterior_survival
sample_prior_survival
sample_posterior_survival
CredibleBand
credible_band
prior_credible_band
posterior_credible_band
RestrictedMeanSurvivalTime
RestrictedMeanSurvivalTimeContrast
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