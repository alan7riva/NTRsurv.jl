"""
    credible_band( p::Float64, S::Matrix{Float64}, μ::Bool=true)

Immutable type for credible bands with lower band extreme `l`, medium band values `m` and upper band extreme `u`.
"""
struct CredibleBand
    t::Vector{Float64}
    d::Vector{Float64}
    m::Vector{Float64}
    u::Vector{Float64}
    s::String
    p::Float64
    draws::Int64
end

"""
    credible_band( p::Float64, S::Matrix{Float64}, μ::Bool=true)

Function for Monte-Carlo computation of (1-p)% survival credible bands and inner survival estimate, either mean survival, default `μ=true`, or 
median survival, alternative `μ=false`, from a matrix `S` with rows provided by samples of survival curves. The output is
a tuple consisting of the lower band envelope, the inner survival estimate, and the upper band envelope in such order.
"""
function credible_band( t::Vector{Float64}, p::Float64, S::Matrix{Float64}; s::String="mean")
    if  !( 0 < p < 1)
        @error "ERROR: p is not between zero and one."
    end 
    l,k = size(S)
    band_m = zeros(k)
    if s[end-3:end] == "mean"
        band_m = vec(mean(S,dims=1))
    else
        band_m = vec(median(S,dims=1))
    end
    m = round( Int, l*p)
    S_band = copy(S)
    for _ in 1:m
        b = zeros(Float64, k)
        for i in 1:k
            b[i] = maximum(S_band[:,i])-minimum(S_band[:,i])
        end
        b_max_index = sortperm(b)[end]
        s_out_index = findmax( abs.( S_band[:,b_max_index] .- median(S_band[:,b_max_index]) ) )[2]
        S_band = S_band[1:end .!= s_out_index, 1:end  ]
    end
    band_u = [ maximum(S_band[:,i]) for i in 1:k]
    band_d = [ minimum(S_band[:,i]) for i in 1:k]
    return CredibleBand( t, band_d, band_m, band_u, s, p, l)
end

"""
    prior_credible_band( p::Float64, l::Int64, t::Vector{Float64}, α::Float64, baseline::Baseline, μ::Bool=true)

Function for Monte-Carlo computation over time-grid `t`. with `l` a priori samples, of (1-p)% credible bands and either mean survival, default `μ=true`, or 
median survival, alternative `μ=false`, from NTR models with variance modulating parameter `α` and `baseline` struct. Output as in `credible_band`.
"""
function prior_credible_band( p::Float64, l::Int64, t::Vector{Float64}, α::Float64, baseline::Baseline; μ::Bool=true)
    S = sample_prior_survival(l,t,α,baseline)
    s = μ ? "Prior mean" : "Prior median"
    return credible_band( t, p, S; s)
end

"""
    posterior_credible_band( p::Float64, l::Int64, t::Vector{Float64}, model::NeutralToTheRightModel, μ::Bool=true)
    posterior_credible_band( p::Float64, l::Int64, t::Vector{Float64}, z_new::Vector{Float64}, model::CoxNeutralToTheRightModel, μ::Bool=true)
    posterior_credible_band( p::Float64, t::Vector{Float64}, z_new::Vector{Float64}, model::CoxNeutralToTheRightFullyBayesianModel, μ::Bool=true)

Function for Monte-Carlo computation over time-grid `t`. with `l` a posteriori samples, of (1-p)% credible bands and either mean 
survival, default `μ=true`, or median survival, alternative `μ=false`, from either a NTR `model` struct or a Cox NTR `model` struct
with fixed covariates `z_new`. Computation for fully Bayesian Cox NTR models does not require a number of Monte-Carlo simulations `l`
as it already should containg a posterior sample of regression coefficients. Output as in `credible_band`.
"""
function posterior_credible_band( p::Float64, l::Int64, t::Vector{Float64}, model::NeutralToTheRightModel; μ::Bool=true)
    S = sample_posterior_survival(l,t,model)
    s = μ ? "posterior mean" : "posterior median"
    return credible_band( t, p, S; s)
end

function posterior_credible_band( p::Float64, l::Int64, t::Vector{Float64}, z_new::Vector{Float64}, model::PluginCoxNeutralToTheRightModel; μ::Bool=true)
    S = sample_posterior_survival(l,t,z_new,model)
    s = μ ? "posterior mean, covariate Z=$( parse.( Float64, _format_number.(z_new) ) )" : "posterior median, covariate Z=$( parse.( Float64, _format_number.(z_new) ) )"
    s = s*", model=plug-in Bayesian"
    return credible_band( t, p, S; s)
end

function posterior_credible_band( p::Float64, l::Int64, t::Vector{Float64}, z_news::Vector{Vector{Float64}}, 
    model::PluginCoxNeutralToTheRightModel; μ::Bool=true, z_ref::Union{Nothing,Vector{Float64}} = nothing)
    s = μ ? ["posterior mean, covariate Z=$( parse.( Float64,_format_number.(z) ) )" for z in z_news] : ["posterior median, covariate Z=$( parse.( Float64, _format_number.(z) ) )" for z in z_news]
    s = [ v*", model=plugin Bayesian" for v in s]
    Smats = sample_posterior_survival(l,t,z_news,model;z_ref)
    return [ credible_band( t, p, Smats[j]; s=s[j]) for j in eachindex(z_news) ]
end

function posterior_credible_band( p::Float64, t::Vector{Float64}, z_new::Vector{Float64}, model::CoxNeutralToTheRightModel; μ::Bool=true)
    S = sample_posterior_survival(t,z_new,model)
    s = μ ? "posterior mean" : "posterior median"
    s = s*", model=fully Bayesian"
    return credible_band( t, p, S; s)
end

function posterior_credible_band( p::Float64, t::Vector{Float64}, z_news::Vector{Vector{Float64}}, 
    model::CoxNeutralToTheRightModel; μ::Bool=true, z_ref::Union{Nothing,Vector{Float64}} = nothing)
    s = μ ? ["posterior mean, covariate Z=$( parse.( Float64, _format_number.(z) ) )" for z in z_news] : ["posterior median, covariate Z=$( parse.( Float64, _format_number.(z) ) )" for z in z_news]
    s = [v*", model=fully Bayesian" for v in s]
    Smats = sample_posterior_survival(t,z_news,model;z_ref)
    return [ credible_band( t, p, Smats[j]; s=s[j]) for j in eachindex(z_news) ]
end