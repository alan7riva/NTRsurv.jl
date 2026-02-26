"""
    DataNTRnorep

An immutable type containing possibly censored to the right observations and 
associated sufficient statistics for NTR model fitting when there are no repetitions 
on the observations.
The type has the following fields:

- `T::Vector{Float64}`: Sorted observation times.
- `δ::Vector{Int64}`: Censoring indicators, 1 if exact observation and 0 otherwise, for sorted observation times `T`.
- `n::Int64`: Number of observations.
- `R₁::Vector{Int64}`: Number of at risk observations after and including \$T_{(j)}\$.
- `R₂::Vector{Int64}`: Number of at risk observations after \$T_{(j)}\$.
"""
struct DataNTRnorep
    T::Vector{Float64}
    δ::Vector{Int64}
    n::Int64
    R₁::Vector{Int64} 
    R₂::Vector{Int64} 
end


function DataNTRnorep(T::Vector{Float64}, δ::Vector{Int64})
    sp = sortperm( T )
    T = T[ sp ]
    n = length(T)
    δ = δ[ sp ]
    e_ind = δ .== 1
    nᵉ = Float64.(e_ind)
    nᶜ = 1.0 .- nᵉ
    Nᵉ = [ cumsum( nᵉ[end:-1:1] )[end:-1:1]; 0]
    Nᶜ = [ cumsum( nᶜ[end:-1:1] )[end:-1:1]; 0]
    R₁ = Nᵉ .+ Nᶜ 
    R₂ = Nᶜ .+ [ Nᵉ[2:end]; 0]
    return DataNTRnorep( T, δ, n, R₁, R₂)
end

"""
    DataNTRrep

An immutable type containing possibly censored to the right observations and 
associated sufficient statistics for NTR model fitting when there are repetitions 
on the observations.
The type has the following fields:

- `T`: Sorted unique observation times.
- `δ`: Exact observation indicator, 1 if at least one exact observation corresponds and 0 otherwise, in `T`.
- `n`: Number of unique observations.
- `nᵉ`: Number of multiplicities for exact observations in `T``.
- `nᶜ`: Number of multiplicities for exact observations in `T``.
- `R₁`: Number of at risk observations after and including \$T_{(j)}\$.
- `R₂`: Number of at risk observations after \$T_{(j)}\$.
"""
struct DataNTRrep
    T::Vector{Float64}
    δ::Vector{Int64}
    n::Int64
    nᵉ::Vector{Int64}
    nᶜ::Vector{Int64}
    R₁::Array{Int64,1}
    R₂::Array{Int64,1}
end

function DataNTRrep(T::Vector{Float64}, δ::Vector{Int64})
    sp = sortperm( T )
    T = T[ sp ]
    δ = δ[ sp ]
    Tu = unique(T)
    n = length(Tu)
    Iᵉ = [ findall( (T .== v) .&& (δ .== 1.0) ) for v in unique(T) ]
    Iᶜ = [ findall( (T .== v) .&& (δ .== 0.0) ) for v in unique(T) ]
    nᵉ = [ length(v) for v in Iᵉ ]
    nᶜ = [ length(v) for v in Iᶜ ]
    δ = 1*( nᵉ .> 0 )
    Nᵉ = [ cumsum( nᵉ[end:-1:1] )[end:-1:1]; 0]
    Nᶜ = [ cumsum( nᶜ[end:-1:1] )[end:-1:1]; 0]
    R₁ = Nᵉ .+ Nᶜ 
    R₂ = Nᶜ .+ [ Nᵉ[2:end]; 0]
    return DataNTRrep( Tu, δ, n, nᵉ, nᶜ, R₁, R₂)
end

"""
    DataNTR

Union type representing survival data objects for possibly censored to the right survival data in NTR models.

`DataNTR` is an alias for the union of internal data objects `DataNTRnorep` and `DataNTRrep`, corresponding respectively to datasets without and 
with repeated event times.
    
    DataNTR(T::Vector{Float64}, δ::Vector{Int64})

Constructor for `DataNTR` with observed event times `T` and censoring indicators `δ` , where `δ[i] = 1` denotes an exact event and
`δ[i] = 0` denotes right censoring.
"""
const DataNTR = Union{DataNTRnorep, DataNTRrep}

function DataNTR(T::Vector{Float64}, δ::Vector{Int64})
    if minimum(T) < 0.0
        @error "Negative values in T are not supported for the data struct!"
    end
    if unique(T) != T
        return DataNTRrep(T, δ)
    else
        return DataNTRnorep(T, δ)
    end
end

"""
    BaselineNTR

Immutable object for baseline specification of NTR prior.

`BaselineNTR` is specified by a cumulative hazard function and, optionally,
its hazard rate and inverse cumulative hazard.

    BaselineNTR(κ::Function)
    BaselineNTR(κ::Function, dκ::Function)
    BaselineNTR(κ::Function, dκ::Function, κinv::Function)

Missing fields are set to zero and must be supplied if required for likelihood evaluation or simulation purposes.

# Fields
- `κ::Function`: A priori cumulative hazard.
- `dκ::Function`: A priori hazard rate. Needed for likelihood evaluations.
- `κinv::Function`: A priori inverse cumulative hazard. Can be needed for simulation purposes outisde of NTRsurv's workflow.
"""
struct BaselineNTR
    κ::Function 
    dκ::Function
    κinv::Function
end

function BaselineNTR(κ::Function)
    return BaselineNTR(κ,zero,zero)
end

function BaselineNTR(κ::Function,dκ::Function)
    return BaselineNTR(κ,dκ,zero)
end

"""
    ExponentialBaseline(λ::Float64)

Construct `BaselineNTR` object corresponding to an exponential baseline
hazard with rate parameter `λ`.
"""
function ExponentialBaseline(λ::Float64)
    r = λ[1]
    return BaselineNTR(z->r*z,z->r,z->z/r)
end

"""
    WeibullBaseline(λ::Float64, k::Float64)

Construct `BaselineNTR` object corresponding to a Weibull baseline
hazard with scale parameter `λ` and shape parameter `k`.
"""
function WeibullBaseline(λ::Float64,k::Float64)
    return BaseLineNTR(z->(z/λ)^k,z->k*z^(k-1)/λ^k,z->(λ*z)^(1/k))
end

"""
    EmpBayesBaseline(data::DataNTR,exact::Bool=false)

Construct `BaselineNTR` object corresponding to an empirically Bayesian exponential baseline
hazard with rate which either matches the mean of all 
observations, default choice with `exact=false`, or only of the exact observations, chosen with`exact=true`.
"""
function EmpBayesBaseline(data::DataNTR,exact::Bool=true)
    if exact
        return ExponentialBaseline(1/mean(data.T[data.δ .== 1]))
    else
        return ExponentialBaseline(1/mean(data.T))
    end
end

function _simulate_prior_survival( t::Array{Float64}, α::Float64, β::Float64, baseline::BaselineNTR)
    κ = baseline.κ
    S = zeros(length(t))
    S[1] = 1.0
    cum = 0.0
    for i in 2:length(t)
        shape = β * (κ(t[i]) - κ(t[i-1]))
        cum += rand(Gamma(shape, 1/α))
        S[i] = exp(-cum)
    end
    return S
end

"""
    prior_sim(t::Array{Float64},α::Float64,baseline::BaselineNTR)

Function for prior simulation, over a grid of positive values `t`, of NTR prior with variance modulating parameter `α` and`baseline` object specification.
Intended for prior elicitation before model setting.` 
"""
function simulate_prior_survival( t::Array{Float64}, α::Float64, baseline::BaselineNTR)
    if !iszero(t[1])
        t = [0.0;t]
    end
    β = 1.0/log(1.0+1.0/α)
    return _simulate_prior_survival( t, α, β, baseline)
end

function simulate_prior_survival( l::Int64, t::Array{Float64}, α::Float64, baseline::BaselineNTR)
    if !iszero(t[1])
        t = [0.0;t]
    end
    β = 1.0/log(1.0+1.0/α)
    S_mat = zeros(Float64, l, length(t))
    for i in 1:l
        S_mat[i,:] = _simulate_prior_survival( t, α, β, baseline)
    end
    return S_mat
end

struct ModelNTRnorep
    α::Float64 
    β::Float64 
    baseline::BaselineNTR
    data::DataNTRnorep
end

struct ModelNTRrep
    α::Float64 
    β::Float64 
    baseline::BaselineNTR
    data::DataNTRrep
end

"""
    ModelNTR

Union type representing NTR models for possibly censored to the right survival data.

`ModelNTR` is an alias for the union of internal structs `ModelNTRnorep` and `ModelNTRrep`, corresponding respectively to modeling of datasets without and 
with repeated event times.
    
    ModelNTR(α::Float64,baseline::BaselineNTR,data::DataNTR
    ModelNTR(α::Float64,data::DataNTR)

Constructor for NTR model with a priori variance modulating parameter `α`, `baseline` object specification, and survival data object `data`. 
If `baseline` is not provided then `EmpBayesBaseline(data::DataNTR,)` is used.
"""
const ModelNTR = Union{ModelNTRnorep, ModelNTRrep}

function ModelNTR(α::Float64,baseline::BaselineNTR,data::DataNTRnorep)
    β = 1.0/log(1.0+1.0/α)
    return ModelNTRnorep( α, β, baseline, data)
end

function ModelNTR( α::Float64, baseline::BaselineNTR, data::DataNTRrep)
    β = 1.0/log(1.0+1.0/α)
    return ModelNTRrep( α, β, baseline, data)
end

function ModelNTR(α::Float64,data::DataNTR)
    baseline = EmpBayesBaseline(data)
    return ModelNTR( α, baseline, data)
end

function postmean_cont_incr(k::Int64,t1::Float64,t2::Float64,model::ModelNTR)
    α = model.α
    β = model.β
    κ = model.baseline.κ
    R₁ = model.data.R₁
    return β*( κ(t2)-κ(t1) )*log( (α+R₁[k])/(α+R₁[k]+1.0) )
end

function postmean_disc_incr_rep(k::Int64,model::ModelNTR)
    α = model.α
    nᵉ = model.data.nᵉ
    R₂ = model.data.R₂
    num = 0.0
    den = 0.0
    nk = nᵉ[k] - 1
    R2k = R₂[k]
    @inbounds for l in 0:nk
        b = binomial(nk, l) * (-1.0)^(l+1)
        num += b * log1p(-1/(R2k + α + l + 2))
        den += b * log1p(-1/(R2k + α + l + 1))
    end
    return log(num/den)
end

function postmean_disc_incr_norep(k::Int64,model::ModelNTR) 
    α = model.α
    R₂ = model.data.R₂
    return log( log( (R₂[k]+α+2.0)/(R₂[k]+α+1.0) )/log( (R₂[k]+α+1.0)/(R₂[k]+α) ) )
end

function postmean_disc_incr(k::Int64,model::ModelNTR)
    nᵉ = model.data.nᵉ
    return ( nᵉ[k] == 1 ) ? postmean_disc_incr_norep(k,model) : postmean_disc_incr_rep(k,model)
end

"""
    mean_posterior_survival(t::Array{Float64},model::ModelNTR)

Function for posterior mean survival curve evaluation in NTR `model` over a grid of positive values `t`.
"""
function mean_posterior_survival(t::Array{Float64},model::ModelNTR)
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
            cont_incr_run += postmean_cont_incr(j,prev,cur,model)
            prev = cur
            S[i] = exp( cont_incr_run + disc_incr_run )
            i += 1
        elseif t[i] > τ[j]
            # survival observation between mesh
            cur = τ[j]
            cont_incr_run += postmean_cont_incr(j,prev,cur,model)
            cur = prev
            if nᵉ[j] >= 1
                disc_incr_run += postmean_disc_incr(j,model)
            end
            j += 1
        else
            # fringe reptition case
            cur = τ[j]
            cont_incr_run += postmean_cont_incr(j,prev,cur,model)
            prev = cur
            if nᵉ[j] >= 1
                disc_incr_run += postmean_disc_incr(j,model)
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
        cont_incr_run += postmean_cont_incr(j,prev,cur,model)
        S[i] = exp( cont_incr_run + disc_incr_run )
        i += 1
        k += 1
    end
    return S
end

"""
   post_fix_locw_GammaNTR_accrej

Function for posterior simulation of weights at fixed locations corresponding to exact observations. 

* `l`: Number of simulaions from the vector of posterior weights.
* `data`: Data struct for NTR models, either type DataNTRnorep or DataNTRrep.
* `α`: Gamma process hyperparameter impacting Variance modulation for NTR survival curves.
"""
function post_fix_locw_GammaNTR_accrej_norep(i::Int64,model::ModelNTR)
    α = model.α
    R₂ = model.data.R₂
    k = α+R₂[i]
    Y = rand(Gamma(1.0,1.0/k))
    logU = log(rand(Uniform()))
    while logU > log(1-exp(-Y)) - log(Y)
        Y = rand(Gamma(1.0,1.0/k))
        logU = log(rand(Uniform()))
    end
    return Y
end

function post_fix_locw_GammaNTR_accrej_rep(i::Int64,model::ModelNTR)
    nᵉ = model.data.nᵉ
    α = model.α
    R₂ = model.data.R₂
    k = α+R₂[i]
    nI = nᵉ[i]
    Y = rand(Gamma(nI,1.0/k))
    logU = log(rand(Uniform()))
    while logU > nI*( log(1-exp(-Y)) -log(Y) )       
        Y = rand(Gamma(nI,1.0/k))
        logU = log(rand(Uniform()))
    end
    return Y
end

function cont_incr(k::Int64,t1::Float64,t2::Float64,model::ModelNTR)
    α = model.α
    β = model.β
    κ = model.baseline.κ
    R₁ = model.data.R₁
    return rand(Gamma( β*(κ(t2) - κ(t1)), 1/(α+R₁[k])))
end

function disc_incr(k::Int64,model::ModelNTR)
    nᵉ = model.data.nᵉ
    return ( nᵉ[k] == 1 ) ? post_fix_locw_GammaNTR_accrej_norep(k,model) : post_fix_locw_GammaNTR_accrej_rep(k,model)
end

function _simulate_posterior_survival(t::Array{Float64},model::ModelNTR)
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

"""
   simulate_posterior_survival

Function for simulation of posterior survival curves, over a grid of positive values `t`, for NTR `model`.
"""
function simulate_posterior_survival( t::Vector{Float64}, model::ModelNTR)
    if !iszero(t[1])
        t = [0.0;t]
    end
    return _simulate_posterior_survival(t, model)
end

function simulate_posterior_survival( l::Int64, t::Vector{Float64}, model::ModelNTR)
    if !iszero(t[1])
        t = [0.0;t]
    end
    S_mat = zeros(Float64, l, length(t))
    for i in 1:l
        S_mat[i,:] = _simulate_posterior_survival( t, model)
    end
    return S_mat
end

"""
    loglikNTR

Function for log-likelihood evaluation of NTR model with a priori variance modulating parameterof `α` 
`baseline` object specification, and survival data object `data`.
"""
function loglikNTR(α::Float64,baseline::BaselineNTR,data::DataNTRnorep)
    l = 0.0
    κ = baseline.κ
    dκ = baseline.dκ
    if dκ == zero
        @error "ERROR: κ derivative not provided in baseline."
    end
    β = 1.0/log(1.0+1.0/α₀)
    n = data.n
    X =  [0.0;data.T]
    R₁ = data.R₁
    R₂ = data.R₂
    δ = data.δ
    cont_incr(k::Int64) = β*( κ(X[k+1])-κ(X[k]) )*log( α/(α + R₁[k]) )
    disc_incr(k::Int64) = log( dκ(X[k+1]) ) + log(β) + log( log( 1.0 + 1.0/(R₂[k]+α) ) ) 
    for k in 1:n
        l += cont_incr(k)
        if δ[k] == 1
            l += disc_incr(k)
        end
    end
    return l
end

function loglikNTR(α::Float64,baseline::BaselineNTR,data::DataNTRrep)
    l = 0.0
    κ = baseline.κ
    dκ = baseline.dκ
    if dκ == zero
        @error "ERROR: κ derivative not provided in baseline."
    end
    β = 1.0/log(1.0+1.0/α₀)
    n = data.n
    nᵉ = [model.data.nᵉ;0]
    X =  [0.0;data.T]
    R₁ = data.R₁
    R₂ = data.R₂
    cont_incr(k::Int64) = β*( κ(X[k+1])-κ(X[k]) )*log( α/(α + R₁[k]) )
    disc_incr(k::Int64) = log( dκ(X[k+1]) ) + log(β) + log( sum( [ binomial(nᵉ[k]-1,l) * (-1.0)^l * log1p( 1/(α+R₂[k]+l) ) for l in 0:(nᵉ[k]-1) ] ) ) 
    for k in 1:n
        l += cont_incr(k)
        if nᵉ[k] >= 1
            l += disc_incr(k)
        end
    end
    return l
end
 