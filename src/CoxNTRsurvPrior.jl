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
    n = length(T)
    sp = sortperm( T )
    T = T[ sp ]
    δ = δ[ sp ]
    Z = Z[ sp ]
    Tu = unique(T)
    m = length(Tu)
    Iᵉ = [ findall( (T .== v) .&& (δ .== 1.0) ) for v in unique(T) ]
    Iᶜ = [ findall( (T .== v) .&& (δ .== 0.0) ) for v in unique(T) ]
    Zᵉ = [ Z[v] for v in Iᵉ ]
    Zᶜ = [ Z[v] for v in Iᶜ ] 
    nᵉ = [ length(v) for v in Iᵉ ]
    nᶜ = [ length(v) for v in Iᶜ ]
    δᵉ = 1*( nᵉ .> 0 )
    return DataRegreNTRrep( T, Tu, δ, δᵉ, Z, Zᵉ, Zᶜ, n, m, nᵉ)
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

"""
    cox_rs

Cox regression risk score.
"""
cox_rs(c::Vector{Float64},x::Vector{Float64}) = exp( c' * x)


"""
   SuffStatsRegreNTR

Function for sufficient statistics in Cox regression NTR model. 

* `c`: Vector of parameters for regression functions.
* `data`: Data struct for Cox regression NTR models, either type DataRegreNTRnorep or DataRegreNTRrep.
* `baseline`: Baseline struct for Cox regression NTR models.
"""
function SuffStatsRegreNTR(c::Vector{Float64},data::DataRegreNTRnorep,g::Function)
    n=data.n
    δ = data.δ
    Z = data.Z
    hᵉ = [ (δ[i]==1) ? g(c,Z[i]) : 0.0 for i in 1:n ] # frequencies of exact bservations
    hᶜ = [ (δ[i]==0) ? g(c,Z[i]) : 0.0 for i in 1:n ] # frequencies of censored observations
    Hᵉ = [ cumsum( hᵉ[end:-1:1] )[end:-1:1]; 0]
    Hᶜ = [ cumsum( hᶜ[end:-1:1] )[end:-1:1]; 0]
    R₁ = Hᵉ .+ Hᶜ 
    R₂ = Hᶜ .+ [ Hᵉ[2:end]; 0]
    return R₁, R₂, hᵉ
end

function SuffStatsRegreNTR(c::Vector{Float64},data::DataRegreNTRrep,g::Function)
    m = data.m
    Zᵉ = [deepcopy(v) for v in data.Zᵉ]
    Zᶜ = [deepcopy(v) for v in data.Zᶜ] 
    hᵉ = zeros(m)
    for i in 1:m
        if !isempty(Zᵉ[i])
            tmp = findmin([ g(c,v) for v in Zᵉ[i] ])
            hᵉ[i] = tmp[1]
            deleteat!( Zᵉ[i], tmp[2] )
        end
    end
    hᵉ_2 = [ sum( [ g(c,v) for v in Zᵉ[i] ], init=0.0) for i in 1:m ] # frequencies of exact bservations
    hᶜ = [ sum( [ g(c,v) for v in Zᶜ[i] ], init=0.0) for i in 1:m ] # frequencies of censored observations
    Hᵉ = [ cumsum( hᵉ_2[end:-1:1] )[end:-1:1]; 0]
    Hᶜ = [ cumsum( hᶜ[end:-1:1] )[end:-1:1]; 0]
    R₁ = Hᵉ .+ Hᶜ 
    R₂ = Hᶜ .+ [ Hᵉ[2:end]; 0]
    F = [ [ [ length(v), sum( [ g(c,z) for z in Zᵉ[k][v]], init=0.0)] for v in collect(subsets(1:length(Zᵉ[k]))) ] for k in 1:m ]
    return R₁, R₂, hᵉ, F
end

"""
   loglikRegreNTR

Function for sufficient statistics in Cox regression NTR model. 

* `c`: Vector of parameters for Cox regression functions.
* `α`: Gamma process hyperparameter impacting Variance modulation for NTR baseline survival.
* `data`: Data struct for Cox regression NTR models, either type DataRegreNTRnorep or DataRegreNTRrep.
* `baseline`: Baseline struct for Cox regression NTR models.
"""
function loglikRegreNTR(c::Vector{Float64},α::Real,baseline::BaselineNTR,g::Function,data::DataRegreNTRnorep)
    l = 0.0
    κ = baseline.κ
    dκ = baseline.dκ
    β = 1.0/log(1.0+1.0/α)
    n = data.n
    X =  [0.0;data.T]
    R₁, R₂, hᵉ = SuffStatsRegreNTR(c,data,g)
    δ = data.δ
    cont_incr(k::Int64) = β*( κ(X[k+1])-κ(X[k]) )*log( α/(α + R₁[k]) )
    disc_incr(k::Int64) = log( dκ(X[k+1]) ) + log(β) + log( log( 1.0 + hᵉ[k]/(R₂[k]+α) ) )
    for k in 1:n
        l += cont_incr(k)
        if δ[k] == 1
            l += disc_incr(k)
        end
    end
    return l
end

function loglikRegreNTR(c::Vector{Float64},α::Real,baseline::BaselineNTR,g::Function,data::DataRegreNTRrep)
    l = 0.0
    κ = baseline.κ
    dκ = baseline.dκ
    β = 1.0/log(1.0+1.0/α)
    m = data.m
    X =  [0.0;data.T]
    R₁, R₂, hᵉ, F = SuffStatsRegreNTR(c,data,g)
    nᵉ = data.nᵉ
    cont_incr(k::Int64) = β*( κ(X[k+1])-κ(X[k]) )*log( α/(α + R₁[k]) )    
    disc_incr(k::Int64) = log( dκ(X[k+1]) ) + log(β) + log( sum( [ (-1.0)^v[1] * log1p(  hᵉ[k]/( α + R₂[k] + v[2]) ) for v in F[k] ] ) )
    for k in 1:m
        l += cont_incr(k)
        if nᵉ[k] > 0
            l += disc_incr(k)
        end
    end
    return l
end

function loglikRegreNTR(c::Vector{Float64},α::Real,baseline::BaselineNTR,data::DataRegreNTR)
    return loglikRegreNTR(c,α,baseline,cox_rs,data)
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
function post_fix_locw_GammaNTR_accrej(z_new::Vector{Float64},l::Int64,model::ModelRegreNTRnorep)
    n = model.data.n
    δ = model.data.δ
    e_bool = δ .== 1 
    m = sum(e_bool)
    W = zeros(m,l) 
    α = model.α
    ν = model.baseline.f(model.c,z_new) 
    R₂ = model.R₂
    hᵉ = model.hᵉ
    i_run = 1
    for i in collect(1:n)[ e_bool ]
        k = (α+R₂[i])/ν
        c = hᵉ[i]/ν
        for j in 1:l
            Y = rand(Gamma(1.0,1.0/k))
            logU = log(rand(Uniform()))
            while logU > log(1-exp(-c*Y)) - log(c*Y)
                Y = rand(Gamma(1.0,1.0/k))
                logU = log(rand(Uniform()))
            end
            W[i_run,j] = Y
        end
        i_run += 1
    end
    return W
end

function post_fix_locw_GammaNTR_accrej(z_new::Vector{Float64},l::Int64,model::ModelRegreNTRrep)
    n = model.data.m
    δ = model.data.δ
    e_bool = δ .== 1 
    m = sum(e_bool)
    W = zeros(m,l) 
    c = model.c
    α = model.α
    f = model.baseline.f
    ν = f(c,z_new)
    logν = log(ν)
    R₂ = model.R₂
    F = model.F
    i_run = 1
    for i in collect(1:n)[ e_bool ]
        k = (α+R₂[i])/ν
        nI = log(length( F[i] ))/log(2)
        logp = sum([ log(f(c,z)) for z in model.data.Zᵉ[i] ])
        for j in 1:l
            Y = rand(Gamma(nI,1.0/k))
            logU = log(rand(Uniform()))
            while logU > sum([ log(1.0 - exp( -f(c,z)*Y/ν)) for z in model.data.Zᵉ[i] ]) -logp  -nI*( log(Y) -logν )        
                Y = rand(Gamma(nI,1.0/k))
                logU = log(rand(Uniform()))
            end
            W[i_run,j] = Y
        end
        i_run += 1
    end
    return W
end

function post_fix_locw_GammaNTR_accrej(ν::Float64,i::Int64,model::ModelRegreNTRnorep)
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

function post_fix_locw_GammaNTR_accrej(ν::Float64,i::Int64,model::ModelRegreNTRrep)
    α = model.α
    f = model.baseline.f
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

"""
   posterior_sim

Function for simulation of posterior survival curves in a grid of values using the analytical distribution of the increments.

* `t`: Time grid where posterior mean survival is evaluated.
* `model`: Model struct for NTR models.
"""
function posterior_sim(t::Vector{Float64},z_new::Vector{Float64},model::ModelRegreNTR)
    if t[1] != 0.0
        t = [0.0;t]
    end
    S = [1.0]
    l = length(t)
    α = model.α
    β = model.β
    ν = model.g(model.c,z_new) 
    κ = model.baseline.κ
    X =  [0.0;model.data.T]
    δ = [model.data.δ;0]
    R₁ = model.R₁
    cont_incr(k::Int64) = exp( -rand(Gamma( β*(κ(X[k]) - κ(X[k-1])), 1/(α+R₁[k]+ν))) )
    cont_incr(k::Int64,t::Float64) = exp( -rand(Gamma( β*(κ(t) - κ(X[k-1])), 1/(α+R₁[k]+ν))) )
    disc_incr(k::Int64) = exp( -post_fix_locw_GammaNTR_accrej(ν,k,model) )
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