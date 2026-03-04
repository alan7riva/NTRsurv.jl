# SPDX-License-Identifier: MIT

module NTRsurv

using Distributions, ProgressMeter
import Statistics: median
import SpecialFunctions: gamma
import IterTools: subsets
ProgressMeter.ijulia_behavior(:append)


include("NTRsurvPrior.jl")
include("CoxNTRsurvPrior.jl")
include("AdaptiveMH.jl")
include("credible_bands.jl")

export
    BaselineNTR,
    DataNTR,
    DataRegreNTR,
    ExponentialBaseline,
    EmpBayesBaseline,
    MCMCchainAcc,
    ModelNTR,
    ModelRegreNTR,
    RobMonMHtune,
    RobMonMHwithinGIBBStune,
    RandWalkMH,
    RandWalkMHwithinGibbs,
    WeibullBaseline,
    loglikNTR,
    loglikRegreNTR,
    mean_posterior_survival,
    prior_credible_band,
    posterior_credible_band,
    simulate_prior_survival,
    simulate_posterior_survival
end