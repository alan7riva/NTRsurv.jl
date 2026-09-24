# SPDX-License-Identifier: MIT

module NTRsurv

using Distributions, ProgressMeter, StatsAPI, StatsBase
import Statistics: median, std
import SpecialFunctions: gamma
import IterTools: subsets
import Base: Tuple
import Plots: plot, plot!, Plot
ProgressMeter.ijulia_behavior(:append)


include("NTRsurvPrior.jl")
include("CoxNTRsurvPrior.jl")
include("AdaptiveMH.jl")
include("credible_bands.jl")
include("model_interface.jl")

export
    Baseline,
    SurvivalData,
    RegressionSurvivalData,
    ExponentialBaseline,
    EmpiricalBayesBaseline,
    NeutralToTheRightModel,
    PluginCoxNeutralToTheRightModel,
    CoxNeutralToTheRightModel,
    WeibullBaseline,
    acceptance_rate,
    loglikelihood,
    mean_posterior_survival,
    credible_band,
    prior_credible_band,
    posterior_credible_band,
    random_walk_mh,
    random_walk_mh_within_gibbs,
    robbins_monro_mh_tune,
    robbins_monro_mh_within_gibbs_tune,
    RestrictedMeanSurvivalTime,
    RestrictedMeanSurvivalTimeContrast,
    sample_prior_survival,
    sample_posterior_survival
end