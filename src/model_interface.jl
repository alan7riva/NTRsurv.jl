function _show_field( io::IO, label::AbstractString, value; width::Int = 20)
    print( io, "\n  ", rpad(label * ":", width), value)
end

_nobs(data::Union{SurvivalDataNoRep,RegressionSurvivalDataNoRep}) = data.n
_nobs(data::Union{SurvivalDataRep,RegressionSurvivalDataRep}) = data.m
_nexact(data::Union{SurvivalData,RegressionSurvivalData}) = Int(sum(data.nᵉ))
_ncensored(data::Union{SurvivalData,RegressionSurvivalData}) = Int(sum(data.nᶜ))
_ndistinct(data::Union{SurvivalData,RegressionSurvivalData}) = data.n
_zndistinct(data::Union{SurvivalData,RegressionSurvivalData}) = length(unique(data.Z))

# SurvivalData compact 2-argument show, used by Array show, print(obj) and repr(obj)
function Base.show( io::IO, data::SurvivalData)
    print( io, "SurvivalData(", "obs.=", _nobs(data), ", exact=", _nexact(data), ", censored=", _ncensored(data), ", distinct=", _ndistinct(data), ")" )
end

# SurvivalData detailed 3-argument show used by display(obj), standalone representation on the REPL.
function Base.show( io::IO, ::MIME"text/plain", data::SurvivalData)
    nobs = _nobs(data)
    nexact = _nexact(data)
    ncensored = _ncensored(data)
    ndistinct = _ndistinct(data)
    exact_percentage = data.n == 0 ? 0.0 : round( 100.0 * nexact / nobs; digits = 1 )
    censoring_percentage = nobs == 0 ? 0.0 : round( 100.0 * ncensored / nobs; digits = 1 )
    distinct_percentage = nobs == 0 ? 0.0 : round( 100.0 * ndistinct / nobs; digits = 1 )
    print( io, "Right-censored survival data")
    _show_field( io, "Observations", nobs)
    _show_field( io, "Exact events", "$(nexact) ($(exact_percentage)%)")
    _show_field( io, "Right-censored", "$(ncensored) ($(censoring_percentage)%)")
    _show_field( io, "Distinct times", "$(ndistinct) ($(distinct_percentage)%)")
end

# Baseline string description
function _baseline_description(baseline::Baseline)
    if !isempty(baseline.s)
        if baseline.s[1:9] == "Empirical"
            return "Empirical "*baseline.s[10:end]
        else 
            return baseline.s
        end
    end
    has_hazard = baseline.dκ !== zero
    if has_hazard 
        return "Baseline(κ,dκ)"
    else
        return "Baseline(κ)"
    end
end


_format_number(x::Real) = string( round( Float64(x); sigdigits = 5 ) )

# Baseline compact 2-argument show, used by Array show, print(obj) and repr(obj)
function Base.show(io::IO, baseline::Baseline)
    print( io, _baseline_description(baseline) )
end

# Baseline detailed 3-argument show used by display(obj), standalone representation on the REPL.
function Base.show( io::IO, ::MIME"text/plain", baseline::Baseline)
        print( io, "Neutral-to-the-right baseline")
    if !isempty(baseline.s)
        _show_field( io, "Specification", _baseline_description(baseline) )
    else 
         _show_field( io, "Specification", "Custom "*_baseline_description(baseline) )
    end
end

# NeutralToTheRightModel compact 2-argument show, used by Array show, print(obj) and repr(obj)
function Base.show(io::IO, model::NeutralToTheRightModel)
    print( io, "NeutralToTheRightModel( α=$(model.α), $(model.baseline), data = $(model.data) )")
end

# NeutralToTheRightModel detailed 3-argument show used by display(obj), standalone representation on the REPL.
function Base.show( io::IO, ::MIME"text/plain", model::NeutralToTheRightModel)
    print(io, "Neutral-to-the-right model")
    _show_field( io, "Gamma process α", model.α)
    _show_field( io, "Center", model.baseline)
    _show_field( io, "Data", model.data)
end

# RegressionSurvivalData compact 2-argument show, used by Array show, print(obj) and repr(obj)
function Base.show( io::IO, data::RegressionSurvivalData)
    print( io, "SurvivalData(", "obs.=", _nobs(data), ", exact=", _nexact(data), ", censored=", _ncensored(data), ", distinct T = ", _ndistinct(data), ", distinct Z = ", _zndistinct(data), ")" )
end

# RegressionSurvivalData detailed 3-argument show used by display(obj), standalone representation on the REPL.
function Base.show( io::IO, ::MIME"text/plain", data::RegressionSurvivalData)
    nobs = _nobs(data)
    nexact = _nexact(data)
    ncensored = _ncensored(data)
    ndistinct = _ndistinct(data)
    zndistinct = _zndistinct(data)
    exact_percentage = data.n == 0 ? 0.0 : round( 100.0 * nexact / nobs; digits = 1 )
    censoring_percentage = nobs == 0 ? 0.0 : round( 100.0 * ncensored / nobs; digits = 1 )
    distinct_percentage = nobs == 0 ? 0.0 : round( 100.0 * ndistinct / nobs; digits = 1 )
    z_distinct_percentage = nobs == 0 ? 0.0 : round( 100.0 * zndistinct / nobs; digits = 1 )
    print( io, "Right-censored survival data with covariates")
    _show_field( io, "Observations", nobs)
    _show_field( io, "Exact events", "$(nexact) ($(exact_percentage)%)")
    _show_field( io, "Right-censored", "$(ncensored) ($(censoring_percentage)%)")
    _show_field( io, "Covariates", "$(length(data.Z[1]))")
    _show_field( io, "Distinct times", "$(ndistinct) ($(distinct_percentage)%)")
    _show_field( io, "Distinct covariates", "$(zndistinct) ($(z_distinct_percentage)%)")
end

# PluginCoxNeutralToTheRightModel compact 2-argument show, used by Array show, print(obj) and repr(obj)
function Base.show(io::IO, model::PluginCoxNeutralToTheRightModel)
    print( io, "PluginCoxNeutralToTheRightModel( coeff=$(model.c), risk score=$(model.custom == 1 ? "custom g" : "Cox"),  α=$(model.α), $(model.baseline), data = $(model.data) )")
end

# PluginCoxNeutralToTheRightModel detailed 3-argument show used by display(obj), standalone representation on the REPL.
function Base.show( io::IO, ::MIME"text/plain", model::PluginCoxNeutralToTheRightModel)
    print( io, "Plugin Cox Neutral-to-the-right model")
    _show_field( io, "Coefficients", _format_number.(model.c))
    if model.custom == true
        _show_field( io, "Risk score", "Custom g")
    else
        _show_field( io, "Risk score", "Cox")
    end
    _show_field( io, "α", model.α)
    _show_field( io, "Center at Z=zeros", model.baseline)
    _show_field( io, "Data", model.data)
end

# CoxNeutralToTheRightModel compact 2-argument show, used by Array show, print(obj) and repr(obj)
function Base.show(io::IO, model::CoxNeutralToTheRightModel)
    print( io, "CoxNeutralToTheRightModel( coeff. draws = $( length(model.c_vec) ), risk score=$(model.custom == 1 ? "custom g" : "Cox"),  α=$(model.α), $(model.baseline), data = $(model.data) )")
end

# CoxNeutralToTheRightModel detailed 3-argument show used by display(obj), standalone representation on the REPL.
function Base.show( io::IO, ::MIME"text/plain", model::CoxNeutralToTheRightModel)
    print( io, "Cox Neutral-to-the-right model")
    _show_field( io, "Coefficient draws", length(model.c_vec))
    if model.custom == true
        _show_field( io, "Risk score", "Custom g")
    else
        _show_field( io, "Risk score", "Cox")
    end
    _show_field( io, "Gamma process α", model.α)
    _show_field( io, "Center at Z=zeros", model.baseline)
    _show_field( io, "Data", model.data)
end

function StatsAPI.nobs(model::Union{NeutralToTheRightModel,PluginCoxNeutralToTheRightModel,CoxNeutralToTheRightModel})
    _nobs(model.data)
end

function StatsAPI.coef(model::PluginCoxNeutralToTheRightModel)
    model.c
end

function StatsAPI.coef(model::CoxNeutralToTheRightModel)
    mean(model.c_vec)
end

function coefdraws(model::CoxNeutralToTheRightModel)
    model.c_vec
end

_default_coefnames(p::Integer) = [ "c[$j]" for j in 1:p ]

function StatsAPI.coefnames(model::PluginCoxNeutralToTheRightModel)
    return _default_coefnames( length(model.c) )
end

function StatsAPI.coefnames(model::CoxNeutralToTheRightModel)
    return _default_coefnames( length(first(model.c_vec)) )
end

function StatsAPI.coeftable(model::CoxNeutralToTheRightModel;names::AbstractVector{<:AbstractString}=StatsAPI.coefnames(model), level = 0.05)
    post_coeff_mean = only(mean(model.c_vec,dims=1))
    post_coeff_sd = std(model.c_vec)
    post_coeff_intervs =  [ [quantile( [s[j] for s in chain_s], 0.5*level ), quantile( [s[j] for s in chain_s], 1 - 0.5*level )] for j in 1:model.data.p ]
    column_names = [  "Posterior mean",
        "Posterior standard deviation",
        "$Interval(100*(1-p))% Credible"]
    StatsBase.CoefTable( hcat( post_coeff_mean, post_coeff_sd, post_coeff_intervs),
        column_names,
        names,
        0,  # No column is a frequentist p-value.
        0   # No column is a frequentist test statistic.
    )
end

# CredibleBand compact 2-argument show, used by Array show, print(obj) and repr(obj)
function Base.show( io::IO, cred_band::CredibleBand)
    print( io, "CredibleBand( level=$(cred_band.p), estimate=$(cred_band.s), draws=$(cred_band.draws))" )
end

# CredibleBand detailed 3-argument show used by display(obj), standalone representation on the REPL.
function Base.show( io::IO, ::MIME"text/plain", cred_band::CredibleBand)
    widths = cred_band.u .- cred_band.d
    print( io, "Credible band")
    _show_field( io, "Credible level", "$(100.0 - 100*cred_band.p)%")
    if !occursin(", covariate ", cred_band.s)
        if occursin("model", cred_band.s)
            s1, s2 = split( cred_band.s, ", model="; limit=2)
            _show_field( io, "Estimate", uppercasefirst(s1) )
            _show_field( io, "Model", uppercasefirst(s2) )
        else
            _show_field( io, "Estimate", uppercasefirst(cred_band.s) )
        end
    else
        s1, s2 = split( cred_band.s, ", covariate "; limit=2)
        _show_field( io, "Estimate", s1)
        s3, s4 = split( s2, ", model="; limit=2)
        _show_field( io, "Covariate", uppercasefirst(s3) )
        _show_field( io, "Model", uppercasefirst(s4) )
    end
    _show_field( io, "Time grid", _format_number(cred_band.t[1])*" to "*_format_number(cred_band.t[end])*" ($(length(cred_band.t)) points)")
    _show_field( io, "Monte-Carlo paths", cred_band.draws)
    _show_field( io, "Median band width", _format_number(median(widths)))
    _show_field( io, "Maximum band width", _format_number(maximum(widths)))
end

# Auxilliary function for default kwargs
function _default_kwargs(cred_band::CredibleBand)
    perc = 100*(1 - cred_band.p)
    return ( ribbon = ( cred_band.m .- cred_band.d, cred_band.u .- cred_band.m), fillalpha = 0.3, xlabel = "\$t\$", ylabel = "\$S(t)\$",label = "$(perc)% credible band", ylim=(0,1), title=cred_band.s, size = (600, 400))
end

# Extend the plot method for CredibleBand struct
function plot(cred_band::CredibleBand; kwargs...)
    default_kwargs =  _default_kwargs(cred_band)
    plot_kwargs = merge( default_kwargs, (; kwargs...) )
    return plot( cred_band.t, cred_band.m; plot_kwargs...)
end

# Add to the current plot.
function plot!( cred_band::CredibleBand; kwargs...)
    default_kwargs =  _default_kwargs(cred_band)
    plot_kwargs = merge( default_kwargs, (; kwargs...) )
    return plot!( cred_band.t, cred_band.m; plot_kwargs...)
end

# Add to a particular plot object.
function plot!( plt::Plot, cred_band::CredibleBand; kwargs...)
    default_kwargs =  _default_kwargs(cred_band)
    plot_kwargs = merge( default_kwargs, (; kwargs...) )
    return plot!( plt, cred_band.t, cred_band.m; plot_kwargs...)
end


# Auxilliary function for numerical integration with trapezoidal rule
function trapz(x::AbstractVector, y::AbstractVector)
    return sum( 0.5 .* (y[1:end-1] .+ y[2:end]) .* diff(x) )
end

# Auxilliary function for RMST compoutation
function rmst_comp(t::Vector{Float64}, S::Matrix{Float64})
    return [ trapz(t, S[i, :]) for i in axes(S, 1) ]
end

"""
    RestrictedMeanSurvivalTime

Struct for `restricted mean survival time` sample where `τ` is the restriction time, `z` is the covariates mean survival ,`grid` 
is the underlpying time grid for simulation, `l_grid` is the length of the `grid`, `v` is a vector container for the sample of 
restricted means, `μ` is the mean of `v`, `p` is a redibility level and `cr_I` is the corresponding credibility interval for `v`.
"""
struct RestrictedMeanSurvivalTime
    s::String
    τ::Float64
    z::Vector{Float64}
    grid::Vector{Float64}
    l_grid::Float64
    v::Vector{Float64}
    μ::Float64
    p::Float64
    cr_I::Vector{Float64}
end

function RestrictedMeanSurvivalTime( l::Int64, t::Vector{Float64}, model::NeutralToTheRightModel,p::Float64=0.05)
    s = "Neutral-to-the-right-model"
    τ = t[end]
    l_t = length(t)
    S = sample_posterior_survival( l, t,  model)
    v =  rmst_comp(t,S)
    μ = mean(v)
    cr_I = quantile( v, [0.5*p,1.0 - 0.5*p])
    return RestrictedMeanSurvivalTime(s,τ,Float64[],t,l_t,v,μ,p,cr_I)
end

function RestrictedMeanSurvivalTime( l::Int64, t::Float64, model::NeutralToTheRightModel, m::Int64=500, p::Float64=0.05)
    s = "Neutral-to-the-right-model"
    grid =  collect(LinRange(0.0,t,m))
    S = sample_posterior_survival( l, grid,  model)
    v = rmst_comp(grid,S)
    μ = mean(v)
    cr_I = quantile( v, [0.5*p,1.0 - 0.5*p])
    return RestrictedMeanSurvivalTime(s,t,Float64[],grid,m,v,μ,p,cr_I)
end

function RestrictedMeanSurvivalTime( l::Int64, t::Vector{Float64}, z::Vector{Float64}, model::PluginCoxNeutralToTheRightModel,p::Float64=0.05)
    s = "Plugin Cox Neutral-to-the-right-model"
    τ = t[end]
    l_t = length(t) 
    S = sample_posterior_survival( l, t, z, model)
    v =  rmst_comp(t,S)
    μ = mean(v)
    cr_I = quantile( v, [0.5*p,1.0 - 0.5*p])
    return RestrictedMeanSurvivalTime(s,τ,z,t,l_t,v,μ,p,cr_I)
end

function RestrictedMeanSurvivalTime( l::Int64, t::Float64, z::Vector{Float64}, model::PluginCoxNeutralToTheRightModel, m::Int64=500,p::Float64=0.05)
    s = "Plugin Cox Neutral-to-the-right-model"
    grid =  collect(LinRange(0.0,t,m))
    S = sample_posterior_survival( l, grid, z, model)
    v = rmst_comp(grid,S)
    μ = mean(v)
    cr_I = quantile( v, [0.5*p,1.0 - 0.5*p])
    return RestrictedMeanSurvivalTime(s,t,z,grid,m,v,μ,p,cr_I)
end

function RestrictedMeanSurvivalTime( t::Vector{Float64}, z::Vector{Float64}, model::CoxNeutralToTheRightModel,p::Float64=0.05)
    s = "Cox Neutral-to-the-right-model"
    τ = t[end]
    l_t = length(t) 
    S = sample_posterior_survival(  t, z, model)
    v =  rmst_comp(t,S)
    μ = mean(v)
    cr_I = quantile( v, [0.5*p,1.0 - 0.5*p])
    return RestrictedMeanSurvivalTime(s,τ,z,t,l_t,v,μ,p,cr_I)
end

function RestrictedMeanSurvivalTime( t::Float64, z::Vector{Float64}, model::CoxNeutralToTheRightModel, m::Int64=500,p::Float64=0.05)
    s = "Cox Neutral-to-the-right-model"
    grid =  collect(LinRange(0.0,t,m))
    S = sample_posterior_survival(  grid, z, model)
    v = rmst_comp(grid,S)
    μ = mean(v)
    cr_I = quantile( v, [0.5*p,1.0 - 0.5*p])
    return RestrictedMeanSurvivalTime(s,t,z,grid,m,v,μ,p,cr_I)
end

# CredibleBand compact 2-argument show, used by Array show, print(obj) and repr(obj)
function Base.show( io::IO, rmst::RestrictedMeanSurvivalTime)
    print( io, "RestrictedMeanSurvivalTime( restriction time = $(rmst.t), mean=$(rmst.μ), credible interval =$(rmst.cr_I), level=$( 100*(1-rmst.p) )%, draws=$(rmst.l_grid)" )
end

# CredibleBand detailed 3-argument show used by display(obj), standalone representation on the REPL.
function Base.show( io::IO, ::MIME"text/plain", rmst::RestrictedMeanSurvivalTime)
    print( io, "Restricted mean survival time")
    _show_field( io, "Restriction time", "$(rmst.τ)")
    if  !isnothing(rmst.z)
        _show_field( io, "Covariate", "$( parse.( Float64, _format_number.(rmst.z)) )")
    end
    _show_field( io, "Model", rmst.s)
    _show_field( io, "Draws", "$(rmst.l_grid)")
    _show_field( io, "Mean", _format_number(rmst.μ))
    _show_field( io, "Credible interval", parse.( Float64, _format_number.(rmst.cr_I)) )
    _show_field( io, "Level", "$( 100*(1-rmst.p) )%")
end

"""
    RestrictedMeanSurvivalTimeContrast

Struct for `restricted mean survival time contrast` sample where `τ` is the restriction time, `z₁` and `z₂` are the covariates for the constrast
related to μ(z₂) - μ(z₁), `grid` is the underlpying time grid for simulation, `l_grid` is the length of the `grid`, `v` is a vector container 
for the sample of contrasts, `μ` is the mean of `v`, `p` is a redibility level and `cr_I` is the corresponding credibility interval for `v`.
"""
struct RestrictedMeanSurvivalTimeContrast
    s::String
    τ::Float64
    z₁::Vector{Float64}
    z₂::Vector{Float64}
    grid::Vector{Float64}
    l_grid::Float64
    v::Vector{Float64}
    μ::Float64
    p::Float64
    cr_I::Vector{Float64}
end

function RestrictedMeanSurvivalTimeContrast( l::Int64, t::Vector{Float64}, z₁::Vector{Float64}, z₂::Vector{Float64}, model::PluginCoxNeutralToTheRightModel, p::Float64=0.05)
    s = "Plugin Cox Neutral-to-the-right-model"
    τ = t[end]
    l_t = length(t) 
    Sv = sample_posterior_survival( l, t, [z₁,z₂], model)
    v = rmst_comp(t, Sv[2] .- Sv[1])
    μ = mean(v)
    cr_I = quantile( v, [0.5*p,1.0 - 0.5*p])
    return RestrictedMeanSurvivalTimeContrast(s,τ,z₁,z₂,t,l_t,v,μ,p,cr_I)
end

function RestrictedMeanSurvivalTimeContrast(l::Int64, t::Float64, z₁::Vector{Float64}, z₂::Vector{Float64}, model::PluginCoxNeutralToTheRightModel, m::Int64=500,p::Float64=0.05)
    s = "Plugin Cox Neutral-to-the-right-model"
    grid =  collect(LinRange(0.0,t,m))
    Sv = sample_posterior_survival( l, grid, [z₁,z₂], model)
    v = rmst_comp(t, Sv[2] .- Sv[1])
    μ = mean(v)
    cr_I = quantile( v, [0.5*p,1.0 - 0.5*p])
    return RestrictedMeanSurvivalTimeContrast(s,τ,z₁,z₂,grid,m,v,μ,p,cr_I)
end

function RestrictedMeanSurvivalTimeContrast( t::Vector{Float64}, z₁::Vector{Float64}, z₂::Vector{Float64}, model::CoxNeutralToTheRightModel, p::Float64=0.05)
    s = "Cox Neutral-to-the-right-model"
    τ = t[end]
    l_t = length(t) 
    Sv = sample_posterior_survival( t, [z₁,z₂], model)
    v = rmst_comp(t, Sv[2] .- Sv[1])
    μ = mean(v)
    cr_I = quantile( v, [0.5*p,1.0 - 0.5*p])
    return RestrictedMeanSurvivalTimeContrast(s,τ,z₁,z₂,t,l_t,v,μ,p,cr_I)
end

function RestrictedMeanSurvivalTimeContrast( t::Float64, z₁::Vector{Float64}, z₂::Vector{Float64}, model::CoxNeutralToTheRightModel, m::Int64=500,p::Float64=0.05)
    s = "Cox Neutral-to-the-right-model"
    grid =  collect(LinRange(0.0,t,m))
    Sv = sample_posterior_survival( grid, [z₁,z₂], model)
    v = rmst_comp(t, Sv[2] .- Sv[1])
    μ = mean(v)
    cr_I = quantile( v, [0.5*p,1.0 - 0.5*p])
    return RestrictedMeanSurvivalTimeContrast(s,τ,z₁,z₂,grid,m,v,μ,p,cr_I) 
end

# CredibleBand compact 2-argument show, used by Array show, print(obj) and repr(obj)
function Base.show( io::IO, rmst::RestrictedMeanSurvivalTimeContrast)
    print( io, "RestrictedMeanSurvivalTimeContrast( restriction time = $(rmst.t), mean=$(rmst.μ), credible interval =$(rmst.cr_I), level=$(rmst.p)), draws=$(rmst.l_grid)" )
end

# CredibleBand detailed 3-argument show used by display(obj), standalone representation on the REPL.
function Base.show( io::IO, ::MIME"text/plain", rmst::RestrictedMeanSurvivalTimeContrast)
    print( io, "Restricted mean survival time contrast")
    _show_field( io, "Restriction time", "$(rmst.τ)")
    _show_field( io, "Model", rmst.s)
    _show_field( io, "Draws", "$(rmst.l_grid)")
    _show_field( io, "Mean", _format_number(rmst.μ))
    _show_field( io, "Credible interval", parse.( Float64, _format_number.(rmst.cr_I) ) ) 
    _show_field( io, "Level", "$(rmst.p)")
end
