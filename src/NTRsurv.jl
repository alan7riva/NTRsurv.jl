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
include("utils.jl")

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
    credible_band,
    loglikNTR,
    loglikRegreNTR,
    mean_posterior_survival,
    simulate_prior_survival,
    simulate_posterior_survival
end