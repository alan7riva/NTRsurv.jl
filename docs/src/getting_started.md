```@meta
CurrentModule = NTRsurv
CollapsedDocStrings = true
```

## Quick start: the Rossi recidivism data

We illustrate the workflow with the Rossi recidivism data. The event-time data contain observed follow-up
times in weeks and indicators of whether a rearrest occurred during the follow-up period.

```@repl rossi
using Random, CSV, DataFrames, NTRsurv, Survival, Statistics, StatsModels, StatsAPI, Plots
Random.seed!(1234);
Rossi = CSV.read( joinpath(pkgdir(NTRsurv), "examples", "data", "rossi.csv"), DataFrame)
```

### Constructing the NTR model

A `SurvivalData` object stores the observed times, censoring indicators, and the
sufficient statistics required by the NTR posterior computations. The `baseline` 
determines the prior mean survival curve, while `α` modulates the prior dispersion.
Here we use an empirical Bayes baseline constructed from the data.


```@repl rossi
T = Float64.(Rossi.week);
δ = Int64.(Rossi.arrest);
data = SurvivalData(T, δ)
```

The sample size, number of observed events, and number of right-censored
observations are:

```@repl rossi
(nrow(Rossi), sum(δ), length(δ) - sum(δ))
```

```@repl rossi
α = 1.0;
baseline = EmpiricalBayesBaseline(data)
```
Assessment of the prior specification can be done by plotting a prior credibility band

```@repl rossi
t = collect(range(0.0, maximum(T), length = 100));
prior_band = prior_credible_band(t,α,baseline; level=0.95, ndraws=3000)
```
The argument `level = 0.95` for the credible band discards five percent of the `ndraws = 3000` sampled paths in the
Monte-Carlo band computation, yielding an approximate 95% credible band. The default values for the keyword arguments are respectively $0.95$ and $3000$, with similar functionality for posterior credible bands as illustrated below.

A method for plotting credible bands has been implemented in `Plots.jl`

```@repl rossi
prior_plot = plot(prior_band);
savefig( prior_plot, "prior.svg"); nothing # hide
```
![](prior.svg)

The neutral to the right model with such baseline prior is constructed with

```@repl rossi
NTR_model = NeutralToTheRightModel(α, baseline, data)
```

### Posterior mean, median and credible band

The posterior mean is available analytically, while credible bands are computed
from posterior survival draws; both are evaluated on a time mesh `t`.

```@repl rossi
posterior_mean = mean_posterior_survival(t, NTR_model);
posterior_band = posterior_credible_band(t, NTR_model; level=0.95, ndraws=3000)
```

The returned CredibleBand struct has fields `d` for the downward envelope, `u` for the upward envelope and `m` for a
middle curve which is the Monte Carlo posterior median by default, with argument `mean_center=false` omitted, 
or the Monte Carlo posterior mean when `mean_center = true`.

For comparison, we also compute the Kaplan--Meier estimator using
[`Survival.jl`](https://juliastats.org/Survival.jl/latest/).

```@repl rossi
km = fit(Survival.KaplanMeier, T, δ);
```
!!! details "Show plotting code"
    ```@repl rossi
    NTR_baseline = exp.(-baseline.κ.(t));
    NTR_plot = plot( posterior_band, label = "NTR posterior median with 95% credible band", title = "Posterior NTR survival curves", size = (720, 430));
    plot!( NTR_plot, t, posterior_mean, linewidth = 2, label = "NTR posterior mean survival (analytic)");
    plot!( NTR_plot, km.events.time, km.survival, seriestype = :steppost, linewidth = 2, linestyle = :dash, label = "Kaplan--Meier estimator");
    plot!( NTR_plot, t, NTR_baseline, linewidth = 2, linestyle = :dot, label = "Prior mean survival (baseline)");
    savefig(NTR_plot, "rossi_ntr_fit.svg"); nothing # hide
    ```

![](rossi_ntr_fit.svg)

Posterior restricted mean survival times,

```math
\Delta_\tau
=
\int_0^\tau
S(t\mid \pmb{\mathcal{D}} )
\,\mathrm{d}t,
```

are computed with

```@repl rossi
rmst =  RestrictedMeanSurvivalTime( t, NTR_model)
```

## Cox-NTR regression

We set a `RegressionSurvivalData` object which stores the observed times, censoring 
indicators, covariates and required sufficient statistics for the Cox-NTR posterior 
computations.

```@repl rossi
z_keys = [:fin, :age, :race, :wexp, :mar, :paro, :prio];
Z = [ [Float64(Rossi[j, key]) for key in z_keys] for j in 1:nrow(Rossi) ];
dataregre = RegressionSurvivalData(T, δ, Z)
```

A practical Cox-NTR workflow first obtains a point estimate of the regression
coefficients and then conditions the Cox-NTR posterior computations on that
estimate. For example, either estimates from an MCMC sampler of the posterior 
distribution or using the frequentist estimator are good alternatives.

```@setup rossi
# Compilation runs
NTRsurv.loglikelihood([0.3,0.1,-0.3,0.3,0.1,-0.3,-0.1],α,baseline,dataregre) # In this notebbok loglikelihood funcion is also loaded by Survival so it has to be sepcified.
robbins_monro_mh_within_gibbs_tune(3,3,x->NTRsurv.loglikelihood(x,α,baseline,dataregre),zeros(7),0.1.*ones(7),0.4.*ones(7),0.7)
```

```@repl rossi
# Robbins-Monro algorithm for variance of proposal distribution tuning
sd_prop_tuned, c₀_tune, llikc₀_tune = robbins_monro_mh_within_gibbs_tune(150,100,x->NTRsurv.loglikelihood(x,α,baseline,dataregre),zeros(7),0.1.*ones(7),0.4.*ones(7),0.7;show_progress=false);
# Metropolis-Hastings within Gibbs chain run
chain_c, _ =  random_walk_mh_within_gibbs( 3000, x-> NTRsurv.loglikelihood(x,α,baseline,dataregre), c₀_tune, llikc₀_tune, sd_prop_tuned[end]);
# Posterior mean estimate of regression coefficients for plugin NTR-Cox model
c_post = mean(chain_c)
```
We set the plugin NTR model using the posterior mean `c_post` computed above.
```@repl rossi
plugin_NTR_Cox_model = PluginCoxNeutralToTheRightModel( c_post, α, baseline, dataregre)
```
We can see that the posterior mean estimator of the Bayesian NTR workflow
for the regression coefficients `c_post` is quite similar to the frequentist 
Cox partial likelihood estimator obtained with the `Survival.jl` package.

```@repl rossi
Rossi.event = Survival.EventTime.(Rossi.week, Rossi.arrest .== 1);
freq_Cox_model = Survival.coxph( @formula(event ~ fin + age + race + wexp + mar + paro + prio), Rossi);
c_freq = coef(freq_Cox_model)
```
We can also fit a fully Bayesian model where posterior uncertainty  on the regression coefficients
is accounted for.

```@repl rossi
NTR_Cox_model = CoxNeutralToTheRightModel( chain_c, α, baseline, dataregre)
```

Furthermore we can use the `coeftable` function from the `StatsAPI.jl` package to compare the
standard deviation estimates for regression coefficients between the frequentist and Bayesian
nonparametric approaches. The frequentist table reports estimated sampling standard errors,
whereas the Bayesian table reports marginal posterior standard deviations and central 
equal-tailed 95% credible interval intervals.

```@repl rossi
StatsAPI.coeftable(freq_Cox_model)
```
and
```@repl rossi
StatsAPI.coeftable( NTR_Cox_model;names=StatsAPI.coefnames(freq_Cox_model))
```

We construct posterior credible bands for two covariate profiles 
`z_1 = [0.0, 23.0, 1.0, 1.0, 0.0, 1.0, 2.0]` and `z_2 = [1.0, 23.0, 1.0, 1.0, 0.0, 1.0, 2.0]`
corresponding to the median population with (`z_2[1]=1.0`) and without (`z_1[1]=0.0`) financial aid; both 
covariate vectors are contained in `z_v = [z_1, z_2]`.
!!! details "Show plotting code"
    ```@repl rossi
    z_1 = [ 0.0, 23.0, 1.0, 1.0, 0.0, 1.0, 2.0 ];
    z_2 = [ 1.0, 23.0, 1.0, 1.0, 0.0, 1.0, 2.0 ];
    z_v = [z_1,z_2];
    NTR_Cox_bands = posterior_credible_band( t, z_v, NTR_Cox_model);
    # Plot for financial aid vs no fiancial aid median survival curves
    NTR_Cox_plot =  plot( NTR_Cox_bands[1], c=4, xlabel="\$t\$", ylabel="\$S(t)\$", fillalpha=0.3, label="No financial aid.");
    plot!( NTR_Cox_bands[2], c=5, fillalpha=0.3,  label="Financial aid", title="Posterior Cox-NTR survival curves\n by financial aid", size = (720, 430));
    savefig(NTR_Cox_plot, "rossi_cox_ntr_fit.svg"); nothing # hide
    ```
![](rossi_cox_ntr_fit.svg)

The plugin model conditions on a fixed estimate of the regression coefficients and therefore does not propagate their posterior uncertainty. 
This can lead to narrower credible bands, particularly when coefficient uncertainty is substantial. Comparing the two approaches helps assess 
its impact on survival inference.
!!! details "Show plotting code"
    ```@repl rossi
    plugin_NTR_Cox_bands = posterior_credible_band( t, z_v, plugin_NTR_Cox_model);
    plugin_NTR_Cox_plot =  plot( plugin_NTR_Cox_bands[1], c=4, xlabel="\$t\$", ylabel="\$S(t)\$", fillalpha=0.3, label="Plugin NTR (no financial aid)");
    plot!( plugin_NTR_Cox_bands[2], c=5, fillalpha=0.3,  label="Plugin NTR (financial aid)", title="Fully Bayesian vs plugin NTR survival curves", size = (720, 430));
    plot!( t, NTR_Cox_bands[1].d, c=:red, linestyle=:dash, fillalpha=0.3,  label="Fully Bayesian NTR (no financial aid)");
    plot!( t, NTR_Cox_bands[1].u, c=:red, linestyle=:dash, fillalpha=0.3,  label="");
    plot!( t, NTR_Cox_bands[2].d, c=:orange, linestyle=:dash, fillalpha=0.3,  label="Fully Bayesian NTR (financial aid)");
    plot!( t, NTR_Cox_bands[2].u, c=:orange, linestyle=:dash, fillalpha=0.3,  label="");
    savefig(plugin_NTR_Cox_plot, "plugin_rossi_cox_ntr_fit_compare.svg"); nothing # hide
    ```
![](plugin_rossi_cox_ntr_fit_compare.svg)

We can also compute posterior restricted mean survival time contrasts, 
```math
\Delta_\tau( \pmb z_{1}, \pmb z_{2})
=
\int_0^\tau
\left\{
S(t\mid \pmb{Z} =  \pmb z_{2})
-
S(t\mid \pmb{Z} =  \pmb z_{1})
\right\}
\,\mathrm{d}t,
```
in Cox-NTR models via

```@repl rossi
rmst_contrast =  RestrictedMeanSurvivalTimeContrast( t, z_1, z_2, NTR_Cox_model)
```