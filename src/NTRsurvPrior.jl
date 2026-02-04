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

"""
    prior_sim(t::Array{Float64},α::Float64,baseline::BaselineNTR)

Function for prior simulation, over a grid of positive values `t`, of NTR prior with variance modulating parameter `α` and`baseline` object specification.
Intended for prior elicitation before model setting.` 
"""
function prior_sim(t::Array{Float64},α::Float64,baseline::BaselineNTR)
    β = 1.0/log(1.0+1.0/α)
    κ = baseline.κ
    if t[1] != 0.0
        t = [0.0;t]
    end
    return [1.0;exp.( -cumsum( [ rand(Gamma(β*(κ(t[i])-κ(t[i-1])),1/α)) for i in 2:length(t) ] ) )]
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

"""
    postmeansurvt::Array{Float64},model::ModelNTR)

Function for posterior mean survival curve evaluation in NTR `model` over a grid of positive values `t`.
"""
function postmeansurv(t::Array{Float64},model::ModelNTRnorep)
    if t[1] != 0.0
        t = [0.0;t]
    end
    S = [1.0]
    l = length(t)
    κ = model.baseline.κ
    α = model.α
    β = model.β
    X =  [0.0;model.data.T]
    δ = model.data.δ
    R₁ = model.data.R₁
    R₂ = model.data.R₂
    cont_incr(k::Int64) = exp( β*( κ(X[k])-κ(X[k-1]) )*log( (α+R₁[k])/(α+R₁[k]+1.0) ) )
    cont_incr(k::Int64,t::Float64) = exp( β*( κ(t)-κ(X[k-1]) )*log( (α+R₁[k])/(α+R₁[k]+1.0) ) )
    disc_incr(k::Int64) = log( (R₂[k]+α+2.0)/(R₂[k]+α+1.0) )/log( (R₂[k]+α+1.0)/(R₂[k]+α) )
    cont_fact_run = 1.0
    n_prev = 1
    disc_fact_run = 1.0
    l_rec = findlast( t .< X[end] )
    for i in 2:l_rec
        X_inc_ind =  t[i-1] .<= X[n_prev+1:end] .< t[i] # indexes of observations which decrease survival between t[i-1] and t[i]
        n_inc = sum(X_inc_ind)
        if n_inc > 0
            n_forw = n_prev + n_inc
            cont_fact_run = cont_fact_run * mapreduce( j -> cont_incr(j),*,(n_prev+1):n_forw,init=1.0) # continuous part factor of decrease running by data observations, no mesh dependence
            disc_fact_run = disc_fact_run * mapreduce( j -> δ[j] == 1 ? disc_incr(j) : 1.0,*,(n_prev+1):n_forw,init=1.0) # discrete part factor of decrease running by data observations, no mesh dependence
            n_prev =  n_forw
        end
        push!( S, cont_fact_run*cont_incr(n_prev+1,t[i]) * disc_fact_run )
    end
    if l_rec < l
        cont_fact_run = cont_fact_run * cont_incr(n_prev+1,t[l_rec])
        if δ[end] ==  1
            disc_fact_run = disc_fact_run*disc_incr(n_prev+1)
        end
        for i in (l_rec+1):l
            push!( S, cont_fact_run*cont_incr(n_prev+1,t[i])*disc_fact_run )
        end
    end
    return S
end

function postmeansurv(t::Array{Float64},model::ModelNTRrep)
    if t[1] != 0.0
        t = [0.0;t]
    end
    S = [1.0]
    l = length(t)
    α = model.α
    β = model.β
    κ = model.baseline.κ
    X =  [0.0;model.data.T]
    nᵉ = [model.data.nᵉ;0]
    R₁ = model.data.R₁
    R₂ = model.data.R₂
    cont_incr(k::Int64) = exp( β*( κ(X[k])-κ(X[k-1]) )*log( (α+R₁[k])/(α+R₁[k]+1.0) ) )
    cont_incr(k::Int64,t::Float64) = exp( β*( κ(t)-κ(X[k-1]) )*log( (α+R₁[k])/(α+R₁[k]+1.0) ) )
    disc_incr_rep(k::Int64) = sum( [ binomial(nᵉ[k]-1,l) * (-1.0)^(l+1) * log1p( -1/(R₂[k]+α+l+2) ) for l in 0:(nᵉ[k]-1) ] )/sum( [ binomial(nᵉ[k]-1,l) * (-1.0)^(l+1) * log1p( -1/(R₂[k]+α+l+1) ) for l in 0:(nᵉ[k]-1) ] )
    disc_incr_norep(k::Int64) = log( (R₂[k]+α+2.0)/(R₂[k]+α+1.0) )/log( (R₂[k]+α+1.0)/(R₂[k]+α) )
    function disc_incr(k::Int64)
        ( nᵉ[k] == 1 ) ? disc_incr_norep(k) : disc_incr_rep(k)
    end
    cont_fact_run = 1.0
    n_prev = 1
    disc_fact_run = 1.0
    l_rec = findlast( t .< X[end] )
    for i in 2:l_rec
        X_inc_ind =  t[i-1] .<= X[n_prev+1:end] .< t[i] # indexes of observations which decrease survival between t[i-1] and t[i]
        n_inc = sum(X_inc_ind)
        if n_inc > 0
            n_forw = n_prev + n_inc
            cont_fact_run = cont_fact_run * mapreduce( j -> cont_incr(j),*,(n_prev+1):n_forw,init=1.0) # continuous part factor of decrease running by data observations, no mesh dependence
            disc_fact_run = disc_fact_run * mapreduce( j -> nᵉ[j] >= 1 ? disc_incr(j) : 1.0,*,(n_prev+1):n_forw,init=1.0) # discrete part factor of decrease running by data observations, no mesh dependence
            n_prev =  n_forw
        end
        push!( S, cont_fact_run*cont_incr(n_prev+1,t[i]) * disc_fact_run )
    end
    if l_rec < l
        cont_fact_run = cont_fact_run * cont_incr(n_prev+1,t[l_rec])
        if nᵉ[end] >=  1
            disc_fact_run = disc_fact_run*disc_incr(n_prev+1)
        end
        for i in (l_rec+1):l
            push!( S, cont_fact_run*cont_incr(n_prev+1,t[i])*disc_fact_run )
        end
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
function post_fix_locw_GammaNTR_accrej(i::Int64,model::ModelNTRnorep)
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

function post_fix_locw_GammaNTR_accrej(i::Int64,model::ModelNTRrep)
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

"""
   posterior_sim

Function for simulation of posterior survival curves, over a grid of positive values `t`, for NTR `model`.
"""
function posterior_sim(t::Array{Float64},model::ModelNTR)
    if t[1] != 0.0
        t = [0.0;t]
    end
    S = [1.0]
    l = length(t)
    α = model.α
    β = model.β
    κ = model.baseline.κ
    X =  [0.0;model.data.T]
    δ = [model.data.δ;0]
    R₁ = model.data.R₁
    cont_incr(k::Int64) = exp( -rand(Gamma( β*(κ(X[k]) - κ(X[k-1])), 1/(α+R₁[k]))) )
    cont_incr(k::Int64,t::Float64) = exp( -rand(Gamma( β*(κ(t) - κ(X[k-1])), 1/(α+R₁[k]))) )
    disc_incr(k::Int64) = exp( -post_fix_locw_GammaNTR_accrej(k,model) )
    cont_fact_run = 1.0
    n_prev = 1
    disc_fact_run = 1.0
    l_rec = findlast( t .< X[end] )
    for i in 2:l_rec
        X_inc_ind =  t[i-1] .<= X[n_prev+1:end] .< t[i] # indexes of observations which decrease survival between t[i-1] and t[i]
        n_inc = sum(X_inc_ind)
        if n_inc > 0
            n_forw = n_prev + n_inc
            cont_fact_run = cont_fact_run * mapreduce( j -> cont_incr(j),*,(n_prev+1):n_forw,init=1.0) # continuous part factor of decrease running by data observations, no mesh dependence
            disc_fact_run = disc_fact_run * mapreduce( j -> δ[j] == 1 ? disc_incr(j) : 1.0,*,(n_prev+1):n_forw,init=1.0) # discrete part factor of decrease running by data observations, no mesh dependence
            n_prev =  n_forw
        end
        push!( S, cont_fact_run*cont_incr(n_prev+1,t[i]) * disc_fact_run )
    end
    if l_rec < l
        cont_fact_run = cont_fact_run * cont_incr(n_prev+1,t[l_rec])
        if δ[end] >=  1
            disc_fact_run = disc_fact_run*disc_incr(n_prev+1)
        end
        for i in (l_rec+1):l
            push!( S, cont_fact_run*cont_incr(n_prev+1,t[i])*disc_fact_run )
        end
    end
    return S
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
    disc_incr(k::Int64) = log( dκ(X[k+1]) ) + log(β) + log( sum( [ binomial(nᵉ[k]-1,l) * (-1.0)^(l+1) * log1p( -1/(R₂[k]+α+l+1) ) for l in 0:(nᵉ[k]-1) ] ) ) 
    for k in 1:n
        l += cont_incr(k)
        if nᵉ[k] >= 1
            l += disc_incr(k)
        end
    end
    return l
end
 