"""
    DataRegreNTRnorep

An immutable type containing possibly censored to the right observations with covariates and 
associated sufficient statistics, not depending on the Cox regression coefficients, for NTR Cox 
model fitting when there are no repetitions on the observations.
The type has the following fields:

- `T`: Sorted observation times.
- `δ`: Censoring indicators, 1 if exact observation and 0 otherwise, for sorted observation times `T`.
- `Z`: Covariates for sorted observation times `T`.
- `n`: Number of observations.
"""
struct DataRegreNTRnorep
    T::Vector{Float64} 
    δ::Vector{Int64}
    Z::Vector{Vector{Float64}} 
    n::Int64 
    nᵉ::Vector{Int64}
end

function DataRegreNTRnorep(T::Vector{Float64}, δ::Vector{Int64}, Z::Vector{Vector{Float64}})
    sp = sortperm( T )
    T = T[ sp ]
    n = length(T)
    nᵉ = Float64.(δ)
    δ = δ[ sp ]
    Z = Z[ sp ]
    return DataRegreNTRnorep( T, δ, Z, n, nᵉ)
end

"""
    DataregreNTRrep

An immutable type containing possibly censored to the right observations with covariates and 
associated sufficient statistics, not depending on the Cox regression coefficients, for NTR Cox 
model fitting when there are no repetitions on the observations.
The type has the following fields:

- `To`: Sorted observation times.
- `T`: Sorted unique observation times.
- `δ`: Censoring indicators, 1 if exact observation and 0 otherwise, for sorted observation times `T`.
- `δᵉ`: Censoring indicators, 1 if exact observation is associated and 0 otherwise, for unique sorted observation times `T`.
- `δᶜ`: Censoring indicators, 1 if exact observation is associated and 0 otherwise, for unique sorted observation times `T`.
- `Z`: Covariates for sorted observation times `T`.
- `Zᵉ`: Covariates for sorted unique observation times `T` which are exactly observed, allowing for multiplicities.
- `Zᶜ`: Covariates for sorted unique observation times `T` which are not exactly observed, allowing for multiplicities.
- `n`: Number of observations.
- `m`: Number of unique observations.
- `nᵉ`: Frequencies of unique exact observations
"""
struct DataRegreNTRrep
    Tr::Vector{Float64}
    T::Vector{Float64}
    δr::Vector{Int64}
    δ::Vector{Int64}
    Z::Vector{Vector{Float64}}
    Zᵉ::Vector{Vector{Vector{Float64}}}
    Zᶜ::Vector{Vector{Vector{Float64}}}
    n::Int64
    m::Int64
    nᵉ::Vector{Int64}
end

function DataRegreNTRrep(T::Vector{Float64}, δ::Vector{Int64}, Z::Vector{Vector{Float64}})
    m = length(T)
    sp = sortperm( T )
    T = T[ sp ]
    δ = δ[ sp ]
    Z = Z[ sp ]
    Tu = unique(T)
    n = length(Tu)
    Iᵉ = [ findall( (T .== v) .&& (δ .== 1.0) ) for v in unique(T) ]
    Iᶜ = [ findall( (T .== v) .&& (δ .== 0.0) ) for v in unique(T) ]
    Zᵉ = [ Z[v] for v in Iᵉ ]
    Zᶜ = [ Z[v] for v in Iᶜ ] 
    nᵉ = [ length(v) for v in Iᵉ ]
    nᶜ = [ length(v) for v in Iᶜ ]
    δᵉ = 1*( nᵉ .> 0 )
    return DataRegreNTRrep( T, Tu, δ, δᵉ, Z, Zᵉ, Zᶜ, m, n, nᵉ)
end

"""
    DataRegreNTR

Union type representing survival data objects for possibly censored to the right survival data with covariates in 
Cox NTR models.

`DataRegreNTR` is an alias for the union of internal data objects `DataRegreNTRnorep` and `DataRegreNTRrep`, corresponding respectively to datasets without and 
with repeated event times.
    
    DataRegreNTR(T::Vector{Float64}, δ::Vector{Int64}, Z::Vector{Vector{Float64}})

Constructor for `DataNTR` with observed event times `T`, censoring indicators `δ` , where `δ[i] = 1` denotes an exact event and
`δ[i] = 0` denotes right censoring, and covariates Z.
"""
const DataRegreNTR = Union{DataRegreNTRnorep, DataRegreNTRrep}

function DataRegreNTR(T::Vector{Float64}, δ::Vector{Int64}, Z::Vector{Vector{Float64}})
    if minimum(T) < 0.0
        @error "Negative values in T are not supported for the data struct!"
    end
    if unique(T) != T
        return DataRegreNTRrep(T, δ, Z)
    else
        return DataRegreNTRnorep(T, δ, Z)
    end
end

function SuffStatsBaselineNTR(baseline::BaselineNTR,data::DataRegreNTR)
    κ = baseline.κ
    dκ = baseline.dκ
    n = data.n
    X =  [0.0;data.T]
    κincs = [ κ(X[k+1])-κ(X[k]) for k in 1:n]
    dκvec = dκ.(data.T)
    return SuffStatsBaselineNTR(κincs,dκvec)
end

"""
    cox_rs

Cox regression risk score.
"""
cox_rs(c::Vector{Float64},x::Vector{Float64}) = exp( c' * x)

struct SuffStatsRegreNTRnorep
    R₁::Vector{Float64}
    R₂::Vector{Float64}
    hᵉ::Vector{Float64}
end

function SuffStatsRegreNTRnorep(c::Vector{Float64},data::DataRegreNTRnorep,g::Function)
    n=data.n
    δ = data.δ
    Z = data.Z
    hᵉ = [ (δ[i]==1) ? g(c,Z[i]) : 0.0 for i in 1:n ] # frequencies of exact bservations
    hᶜ = [ (δ[i]==0) ? g(c,Z[i]) : 0.0 for i in 1:n ] # frequencies of censored observations
    Hᵉ = [ cumsum( hᵉ[end:-1:1] )[end:-1:1]; 0]
    Hᶜ = [ cumsum( hᶜ[end:-1:1] )[end:-1:1]; 0]
    R₁ = Hᵉ .+ Hᶜ 
    R₂ = Hᶜ .+ [ Hᵉ[2:end]; 0]
    return SuffStatsRegreNTRnorep(R₁, R₂, hᵉ)
end

struct SuffStatsRegreNTRrep
    R₁::Vector{Float64}
    R₂::Vector{Float64}
    hᵉ::Vector{Float64}
    F::Vector{Vector{Vector{Float64}}}
end

function SuffStatsRegreNTRrep(c::Vector{Float64},data::DataRegreNTRrep,g::Function)
    n = data.n
    Zᵉ = [deepcopy(v) for v in data.Zᵉ]
    Zᶜ = [deepcopy(v) for v in data.Zᶜ] 
    hᵉ = zeros(n)
    for i in 1:n
        if !isempty(Zᵉ[i])
            tmp = findmin([ g(c,v) for v in Zᵉ[i] ])
            hᵉ[i] = tmp[1]
            deleteat!( Zᵉ[i], tmp[2] )
        end
    end
    hᵉ_2 = [ sum( [ g(c,v) for v in Zᵉ[i] ], init=0.0) for i in 1:n ] # frequencies of exact bservations
    hᶜ = [ sum( [ g(c,v) for v in Zᶜ[i] ], init=0.0) for i in 1:n ] # frequencies of censored observations
    Hᵉ = [ cumsum( hᵉ_2[end:-1:1] )[end:-1:1]; 0]
    Hᶜ = [ cumsum( hᶜ[end:-1:1] )[end:-1:1]; 0]
    R₁ = Hᵉ .+ Hᶜ 
    R₂ = Hᶜ .+ [ Hᵉ[2:end]; 0]
    F = [ [ [ length(v), sum( [ g(c,z) for z in Zᵉ[k][v]], init=0.0)] for v in collect(subsets(1:length(Zᵉ[k]))) ] for k in 1:n ]
    return SuffStatsRegreNTRrep(R₁, R₂, hᵉ, F)
end

"""
   SuffStatsRegreNTR

Function for sufficient statistics in Cox regression NTR model. 

* `c`: Vector of parameters for regression functions.
* `data`: Data struct for Cox regression NTR models, either type DataRegreNTRnorep or DataRegreNTRrep.
* `baseline`: Baseline struct for Cox regression NTR models.
"""
const SuffStatsRegreNTR = Union{SuffStatsRegreNTRnorep, SuffStatsRegreNTRrep}

function SuffStatsRegreNTR(c::Vector{Float64},data::DataRegreNTR,g::Function)
    if isa(data, DataRegreNTRnorep)
        return SuffStatsRegreNTRnorep(c,data,g)
    else
        return SuffStatsRegreNTRrep(c,data,g)
    end
end

function ll_cont_incr(k::Int64,α::Float64,β::Float64,suffstatsb::SuffStatsBaselineNTR,suffstatsr::SuffStatsRegreNTR)
    κinc = suffstatsb.κincs[k]
    R1k = suffstatsr.R₁[k]
    return β*κinc*log( α/(α + R1k) )
end

function ll_disc_incr_norep(k::Int64,α::Float64,β::Float64,suffstatsb::SuffStatsBaselineNTR,suffstatsr::SuffStatsRegreNTR)
    dκk = suffstatsb.dκvec[k]
    hek = suffstatsr.hᵉ[k]
    R2k = suffstatsr.R₂[k]
    return log( dκk ) + log(β) + log( log( 1.0 + hek/(R2k+α) ) )
end

function ll_disc_incr_rep(k::Int64,α::Float64,β::Float64,suffstatsb::SuffStatsBaselineNTR,suffstatsr::SuffStatsRegreNTR)
    dκk = suffstatsb[k]
    hek = suffstatsr.hᵉ[k]
    R2k = suffstatsr.R₂[k]
    s = 0.0
    for v in F[k]
        s += (-1.0)^v[1] * log1p(  hek/( α + R2k + v[2]) )
    end
    return  log( dκk ) + log(β) + log( s )
end

function ll_disc_incr(k::Int64,α::Float64,β::Float64,nᵉ::Vector{Int64},suffstatsb::SuffStatsBaselineNTR,suffstatsr::SuffStatsRegreNTR) 
    return ( nᵉ[k] == 1 ) ? ll_disc_incr_norep(k,α,β,suffstatsb,suffstatsr) : ll_disc_incr_rep(k,α,β,suffstatsb,suffstatsr)
end

"""
   loglikRegreNTR

Function for sufficient statistics in Cox regression NTR model. 

* `c`: Vector of parameters for Cox regression functions.
* `α`: Gamma process hyperparameter impacting Variance modulation for NTR baseline survival.
* `data`: Data struct for Cox regression NTR models, either type DataRegreNTRnorep or DataRegreNTRrep.
* `baseline`: Baseline struct for Cox regression NTR models.
"""
function loglikRegreNTR(c::Vector{Float64},α::Real,β::Real,suffstatsb::SuffStatsBaselineNTR,g::Function,data::DataRegreNTR)
    l = 0.0
    suffstatsr = SuffStatsRegreNTR(c,data,g)
    n = data.n
    δ = data.δ
    nᵉ = data.nᵉ
    for k in 1:n
        l += ll_cont_incr(k,α,β,suffstatsb,suffstatsr)
        if δ[k] == 1
            l += ll_disc_incr(k,α,β,nᵉ,suffstatsb,suffstatsr)
        end
    end
    return l
end

function loglikRegreNTR(c::Vector{Float64},α::Real,β::Real,baseline::BaselineNTR,g::Function,data::DataRegreNTR)
    suffstatsb = SuffStatsBaselineNTR(baseline,data)
    return loglikRegreNTR(c,α,β,suffstatsb,g,data)
end

function loglikRegreNTR(c::Vector{Float64},α::Real,baseline::BaselineNTR,g::Function,data::DataRegreNTR)
    β = 1.0/log(1.0+1.0/α)
    return loglikRegreNTR(c,α,β,baseline,g,data)
end

function loglikRegreNTR(c::Vector{Float64},α::Real,baseline::BaselineNTR,data::DataRegreNTR)
    return loglikRegreNTR(c,α,baseline,cox_rs,data)
end

function loglikRegreNTR(c::Vector{Float64},α::Real,β::Real,suffstatsb::SuffStatsBaselineNTR,data::DataRegreNTR)
    return loglikRegreNTR(c,α,β,suffstatsb,cox_rs,data)
end

"""
    NTRmodelRegre

An immutable type for the NTR model framweork 
- `data`: Data struct with no repetitions in the obsevrations.
- `baseline`: Baseline struct for Cox regression NTR models.
- `c`: Vector of parameters for Cox regression functions.
- `α`: Gamma process hyperparameter impacting Variance modulation for NTR survival curves.
- `β`: Gamma process hyperparameter chosen for centering of NTR survival curves on baseline.
- `R₁`: Sufficient statistic for number of at risk observations after and including T_{(j)} factors.
- `R₂`: Sufficient statistic for number of at risk observations after T_{(j)} factors.
- `hᵉ`: Sufficient statistic for exact observation covariate factors.
"""

struct ModelRegreNTRnorep
    c::Vector{Float64}
    α::Float64 
    β::Float64
    baseline::BaselineNTR
    g::Function
    data::DataRegreNTRnorep 
    R₁::Vector{Float64}
    R₂::Vector{Float64}
    hᵉ::Vector{Float64}
end

struct ModelRegreNTRrep
    c::Vector{Float64}
    α::Float64
    β::Float64
    baseline::BaselineNTR
    g::Function
    data::DataRegreNTRrep
    R₁::Vector{Float64}
    R₂::Vector{Float64}
    hᵉ::Vector{Float64}
    F::Vector{Vector{Vector{Float64}}}
end

"""
    ModelRegreNTR

Union type representing Cox NTR models for possibly censored to the right survival data with covariates.

`ModelRegreNTR` is an alias for the union of internal structs `ModelRegreNTRnorep` and `ModelRegreNTRrep`, corresponding respectively to modeling of datasets without and 
with repeated event times.
    
    ModelRegreNTR(b::Vector{Float64},α::Float64,baseline::BaselineRegreNTR,data::DataRegreNTR)
    ModelRegreNTR(α::Float64,data::DataNTR)

Constructor for NTR model with a priori variance modulating parameter `α`, `baseline` object specification, and survival data object `data`. 
If `baseline` is not provided then `EmpBayesBaseline(data::DataNTR,)` is used.
"""
const ModelRegreNTR = Union{ModelRegreNTRnorep, ModelRegreNTRrep}

function ModelRegreNTR(c::Vector{Float64},α::Float64,baseline::BaselineNTR,g::Function,data::DataRegreNTRnorep)
    β = 1.0/log(1.0+1.0/α)
    s1, s2, s3 = SuffStatsRegreNTR(c,data,g)
    return ModelRegreNTRnorep( c, α, β, baseline, g, data, s1, s2, s3)
end

function ModelRegreNTR(c::Vector{Float64},α::Float64,baseline::BaselineNTR,g::Function,data::DataRegreNTRrep)
    β = 1.0/log(1.0+1.0/α)
    s1, s2, s3, s4 = SuffStatsRegreNTR(c,data,g)
    return ModelRegreNTRrep( c, α, β, baseline, g, data, s1, s2, s3, s4)
end

function ModelRegreNTR(c::Vector{Float64},α::Float64,baseline::BaselineNTR,data::DataRegreNTR)
    return ModelRegreNTR( c, α, baseline, cox_rs, data)
end

function postmean_cont_incr(k::Int64,t1::Float64,t2::Float64,z_new::Vector{Float64},model::ModelRegreNTR)
    α = model.α
    β = model.β
    c = model.c
    ν = model.g(model.c,z_new) 
    κ = model.baseline.κ
    R₁ = model.R₁
    return β*( κ(t2)-κ(t1) )*log( (α+R₁[k])/(α+R₁[k]+ν) )
end

function postmean_disc_incr_rep(k::Int64,z_new::Vector{Float64},model::ModelRegreNTR)
    α = model.α
    c = model.c
    ν = model.g(model.c,z_new)
    hᵉ = model.hᵉ
    F = model.F
    nᵉ = model.data.nᵉ
    R₂ = model.R₂
    num = 0.0
    den = 0.0
    hk = hᵉ[k]
    Fk = F[k]
    R2k = R₂[k]
    @inbounds for v in Fk
        num += (-1.0)^(v[1]+1) * log( ( α + R2k + ν + hk + v[2])/( α + R2k + ν + v[2]  ) )
        den += (-1.0)^(v[1]+1) * log( ( α + R2k + hk + v[2])/( α + R2k + v[2] ) )
    end
    return log(num/den)
end

function postmean_disc_incr_norep(k::Int64,z_new::Vector{Float64},model::ModelRegreNTR) 
    α = model.α
    c = model.c
    ν = model.g(model.c,z_new)
    hᵉ = model.hᵉ
    R₂ = model.R₂
    return log( log( (R₂[k]+α+ν+hᵉ[k])/(R₂[k]+α+ν) )/log( (R₂[k]+α+hᵉ[k])/(R₂[k]+α) ) )
end

function postmean_disc_incr(k::Int64,z_new::Vector{Float64},model::ModelRegreNTR)
    nᵉ = model.data.nᵉ
    ν = model.g(model.c,z_new) 
    return ( nᵉ[k] == 1 ) ? postmean_disc_incr_norep(k,z_new,model) : postmean_disc_incr_rep(k,z_new,model)
end

"""
    mean_posterior_survival

Function for posterior mean survival curve evaluation over a grid

* `t`: Time grid where posterior mean survival is evaluated.
* `data`: Data struct for NTR models, either type DataNTRnorep or DataNTRrep.
* `baseline`: Baseline struct for NTR models.
* `α`: Gamma process hyperparameter impacting Variance modulation for NTR survival curves.
* `β`: Gamma process hyperparameter chosen for centering of NTR survival curves on baseline.
"""
function mean_posterior_survival(t::Array{Float64}, z_new::Vector{Float64}, model::ModelRegreNTR)
    if t[1] != 0.0
        t = [0.0;t]
    end
    nᵉ = model.data.nᵉ
    τ = model.data.T
    m = length(t)
    n = length(τ)
    S = Vector{eltype(t)}(undef, m)
    S[1] = 1.0
    # Logarithmic scale for numerical stability
    cont_incr_run = 0.0
    disc_incr_run = 0.0
    i = 2
    j = 1
    prev = 0.0
    k = 2
    @inbounds while i ≤ m && j ≤ n
        if t[i] < τ[j]
            # no survival observation between mesh
            cur = t[i]
            cont_incr_run += postmean_cont_incr(j,prev,cur,z_new,model)
            prev = cur
            S[i] = exp( cont_incr_run + disc_incr_run )
            i += 1
        elseif t[i] > τ[j]
            # survival observation between mesh
            cur = τ[j]
            cont_incr_run += postmean_cont_incr(j,prev,cur,z_new,model)
            cur = prev
            if nᵉ[j] >= 1
                disc_incr_run += postmean_disc_incr(j,z_new,model)
            end
            j += 1
        else
            # fringe reptition case
            cur = τ[j]
            cont_incr_run += postmean_cont_incr(j,prev,cur,z_new,model)
            prev = cur
            if nᵉ[j] >= 1
                disc_incr_run += postmean_disc_incr(j,z_new,model)
            end
            S[i] = exp( cont_incr_run + disc_incr_run)
            i += 1
            j += 1
        end
        k += 1
    end
    # last survival observation greater than mesh's end
    @inbounds while i ≤ m
        cur = t[i]
        cont_incr_run += postmean_cont_incr(j,prev,cur,z_new,model)
        S[i] = exp( cont_incr_run + disc_incr_run )
        i += 1
        k += 1
    end
    return S
end

"""
   post_fix_locw_GammaNTR_accrej

Function for posterior simulation of weights at fixed locations corresponding to exact observations. 

- `l`: Number of simulaions from the vector of posterior weights.
- `data`: Data struct for NTR models, either type DataNTRnorep or DataNTRrep.
- `α`: Gamma process hyperparameter impacting Variance modulation for NTR survival curves.
"""
function post_fix_locw_GammaNTR_accrej_norep(ν::Float64,i::Int64,model::ModelRegreNTR)
    α = model.α
    R₂ = model.R₂
    hᵉ = model.hᵉ
    k = (α+R₂[i])/ν
    c = hᵉ[i]/ν
    Y = rand(Gamma(1.0,1.0/k))
    logU = log(rand(Uniform()))
    while logU > log(1-exp(-c*Y)) - log(c*Y)
        Y = rand(Gamma(1.0,1.0/k))
        logU = log(rand(Uniform()))
    end
    return Y
end

function post_fix_locw_GammaNTR_accrej_rep(ν::Float64,i::Int64,model::ModelRegreNTR)
    α = model.α
    g = model.baseline.g
    c = model.c
    logν = log(ν)
    R₂ = model.R₂
    F = model.F
    k = (α+R₂[i])/ν
    nI = log(length( F[i] ))/log(2)
    logp = sum([ log(f(c,z)) for z in model.data.Zᵉ[i] ])
    Y = rand(Gamma(nI,1.0/k))
    logU = log(rand(Uniform()))
    while logU > sum([ log(1.0 - exp( -f(c,z)*Y/ν)) for z in model.data.Zᵉ[i] ]) -logp  -nI*( log(Y) -logν )        
        Y = rand(Gamma(nI,1.0/k))
        logU = log(rand(Uniform()))
    end
    return Y
end

function cont_incr(ν::Float64,k::Int64,t1::Float64,t2::Float64,model::ModelRegreNTR)
    α = model.α
    β = model.β
    κ = model.baseline.κ
    R₁ = model.data.R₁
    return rand(Gamma( β*(κ(t2) - κ(t1)), 1/(α+R₁[k]+ν)))
end

function disc_incr(ν::Float64,k::Int64,model::ModelNTR)
    nᵉ = model.data.nᵉ
    return ( nᵉ[k] == 1 ) ? post_fix_locw_GammaNTR_accrej_norep(ν,k,model) : post_fix_locw_GammaNTR_accrej_rep(ν,k,model)
end

"""
   posterior_sim

Function for simulation of posterior survival curves in a grid of values using the analytical distribution of the increments.

* `t`: Time grid where posterior mean survival is evaluated.
* `model`: Model struct for NTR models.
"""
function _simulate_posterior_survival(t::Array{Float64},z_new::Vector{Float64},model::ModelRegreNTR)
    nᵉ = model.data.nᵉ
    τ = model.data.T
    ν = model.g(model.c,z_new) 
    m = length(t)
    n = length(τ)
    S = Vector{eltype(t)}(undef, m)
    S[1] = 1.0
    # Logarithmic scale for numerical stability
    cont_incr_run = 0.0
    disc_incr_run = 0.0
    i = 2
    j = 1
    prev = 0.0
    k = 2
    @inbounds while i ≤ m && j ≤ n
        if t[i] < τ[j]
            # no survival observation between mesh
            cur = t[i]
            cont_incr_run += cont_incr(j,prev,cur,model)
            prev = cur
            S[i] = exp( -cont_incr_run - disc_incr_run )
            i += 1
        elseif t[i] > τ[j]
            # survival observation between mesh
            cur = τ[j]
            cont_incr_run += cont_incr(j,prev,cur,model)
            cur = prev
            if nᵉ[j] >= 1
                disc_incr_run += disc_incr(j,model)
            end
            j += 1
        else
            # fringe reptition case
            cur = τ[j]
            cont_incr_run += cont_incr(j,prev,cur,model)
            prev = cur
            if nᵉ[j] >= 1
                disc_incr_run += disc_incr(j,model)
            end
            S[i] = exp( - cont_incr_run - disc_incr_run)
            i += 1
            j += 1
        end
        k += 1
    end
    # last survival observation greater than mesh's end
    @inbounds while i ≤ m
        cur = t[i]
        cont_incr_run += cont_incr(j,prev,cur,model)
        S[i] = exp( -cont_incr_run - disc_incr_run )
        i += 1
        k += 1
    end
    return S
end