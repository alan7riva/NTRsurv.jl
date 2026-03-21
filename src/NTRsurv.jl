# SPDX-License-Identifier: MIT

module NTRsurv

using Distributions, ProgressMeter
import Statistics: median
import SpecialFunctions: gamma
import IterTools: subsets
import Base: Tuple
ProgressMeter.ijulia_behavior(:append)


include("NTRsurvPrior.jl")
include("CoxNTRsurvPrior.jl")
include("AdaptiveMH.jl")
include("credible_bands.jl")

export
    Baseline,
    SurvivalData,
    RegressionSurvivalData,
    ExponentialBaseline,
    EmpiricalBayesBaseline,
    NeutralToTheRightModel,
    CoxNeutralToTheRightFullyBayesianModel,
    CoxNeutralToTheRightModel,
    WeibullBaseline,
    acceptance_rate,
    loglikelihood,
    mean_posterior_survival,
    prior_credible_band,
    posterior_credible_band,
    random_walk_mh,
    random_walk_mh_within_gibbs,
    robbins_monro_mh_tune,
    robbins_monro_mh_within_gibbs_tune,
    sample_prior_survival,
    sample_posterior_survival
end